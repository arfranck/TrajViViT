import torch
import torch.nn.functional as F
from torch import nn
import math
from einops import rearrange
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def posemb_sincos_3d(patches, temperature = 10000, dtype = torch.float32):
    _, f, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    z, y, x = torch.meshgrid(
        torch.arange(f, device=device),
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing='ij')

    fourier_dim = dim // 6

    omega = torch.arange(fourier_dim, device = device) / (fourier_dim - 1)
    omega = 1. / (temperature ** omega)

    z = z.flatten()[:, None] * omega[None, :]
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]

    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos(), z.sin(), z.cos()), dim = 1)

    pe = F.pad(pe, (0, dim - (fourier_dim * 6))) # pad if feature dimension not cleanly divisible by 6
    return pe.type(dtype)


# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim if out_dim else dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head),
                FeedForward(dim, mlp_dim)
            ]))


    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# https://discuss.pytorch.org/t/how-to-modify-the-positional-encoding-in-torch-nn-transformer/104308/2
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class SimpleViT(nn.Module):

    def __init__(self, *, image_size=(64, 64), image_patch_size=(16, 16), frames=8, frame_patch_size=1, dim, depth=6, heads=8, mlp_dim=1024,device, channels=1, dim_head=64):
        super().__init__()

        self.device = device
        self.dim = dim
        self.nheads = heads

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, f'Frames must be divisible by the frame patch size {frames} {frame_patch_size}'

        patch_dim = channels * patch_height * patch_width * frame_patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b (f pf) (h p1) (w p2) -> b f h w (p1 p2 pf)', p1=patch_height, p2=patch_width, pf=frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        ).to(self.device)
        
        self.pe = PositionalEncoding(dim).to(self.device)

        self.encoderLayer = nn.TransformerEncoderLayer(dim,nhead=heads,dim_feedforward=mlp_dim, batch_first=True).to(self.device)
        self.encoder = nn.TransformerEncoder(self.encoderLayer,num_layers=depth).to(self.device)
        self.myNet = nn.Linear(dim,2).to(self.device)

        self.decoderLayer = nn.TransformerDecoderLayer(d_model=dim,nhead=heads,batch_first=True).to(self.device)
        self.decoder = nn.TransformerDecoder(self.decoderLayer,depth).to(self.device)

        self.tgt_linear = nn.Linear(2,dim).to(device)


    def forward(self, video,tgt,train=True):
        *_, h, w, dtype = *video.shape, video.dtype
        video = video.to(self.device)


        x = self.to_patch_embedding(video)

        pe = posemb_sincos_3d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.encoder(x)

        x = self.generate_sequence(tgt,x,train)
        output = x

        x = self.myNet(x[:, :-1, :])

        if train:
            return x
        else:
            return x,output

    def generate_sequence(self,tgt, memory,train):
        # Initialize the decoder input with a special start-of-sequence token



        if tgt is not None:

            if train:

                tgt = self.tgt_linear(tgt)

            sos = torch.ones(memory.shape[0], 1, self.dim).to(self.device)
            tgt = torch.cat([sos, tgt], dim=1)
        else:
            tgt = torch.ones(memory.shape[0], 2, self.dim).to(self.device)

        mask = torch.ones((tgt.shape[0]*self.nheads, tgt.shape[1], tgt.shape[1])).to(self.device)
        mask = mask.masked_fill(torch.tril(torch.ones((tgt.shape[1], tgt.shape[1])).to(self.device)) == 0, float('-inf'))
        tgt = self.pe(tgt)
        output = self.decoder(tgt=tgt,memory=memory,tgt_mask=mask)

        return output












