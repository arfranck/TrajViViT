import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import dataset
from torchvision import transforms
import random
import numpy as np
from PIL import Image

"""
    This file is used to generate the data to train, test and validate the TrajViVit model in a 
    sliding window manner. 
"""
class TrajDataset(dataset.Dataset):
    to_tensor = transforms.ToTensor()

    def __init__(self, data_folders, n_prev, n_next, img_step, prop, part=0, limit=None, path=False):

        self.data_folders = data_folders
        self.n_prev = n_prev
        self.n_next = n_next
        self.img_step = img_step
        self.block_size = int(data_folders[0].split("_")[-1])

        self.path = path

        src = []
        tgt = []
        coords = []
        path = [] if path else False

        for folder in self.data_folders:
            rand = random.Random(42)
            raw_data = pd.read_csv(folder + "/annotations_" + str(self.img_step) + ".txt", sep=" ")
            print(len(raw_data["track_id"].unique()))
            all_ids = raw_data[raw_data['occluded'] != 1]["track_id"].unique()
            rand.shuffle(all_ids)
            split_index = (len(all_ids) * np.cumsum(prop)).astype(int)
            trajs_index = np.split(all_ids, split_index[:-1])[part]
            track_ids = raw_data["track_id"].unique()[trajs_index][:limit]

            for track_id in track_ids:
                print("opening track " + str(track_id) + " from " + folder)
                traj = raw_data[raw_data["track_id"] == track_id]  # get all positions of track
                memo = {}
                for i in range(len(traj) - self.n_next - self.n_prev):
                    if path != False:
                        path.append(folder)
                    # n_prev images used to predict
                    x = self.get_n_images_after_i(folder, traj, self.n_prev, i, memo)
                    src.append(x)
                    # coords of the previous images
                    c = traj.iloc[i: i + self.n_prev][["x", "y"]]
                    coords.append(Tensor(c.values))
                    # images that should be predicted
                    y = traj.iloc[i + self.n_prev: i + self.n_prev + self.n_next][
                        ["x", "y"]]  # recuperer le grand truth à prédire
                    tgt.append(Tensor(y.values))  # add to ground truth dataset
        self.src = torch.stack(src, dim=0)
        self.coords = self.normalize_coords(torch.stack(coords, dim=0))
        self.tgt = self.normalize_coords(torch.stack(tgt, dim=0))
        self.path = path

    def normalize_coords(self, tgt):
        print(self.get_image_size()[0])
        return tgt / self.get_image_size()[0]

    def get_n_images_after_i(self, folder, traj, n, i, memo):
        X = []
        for ind, pos in traj.iloc[i: i + n, :].iterrows():
            track_id = pos["track_id"]
            frame = pos["frame"]
            path = f"{folder}/{track_id:03d}_{frame:05d}.jpg"
            if path in memo:
                img = memo[path]
            else:
                img = Image.open(f"{folder}/{track_id:03d}_{frame:05d}.jpg")
                memo[path] = img
            img_tensor = self.to_tensor(img)
            X.append(img_tensor)
        return torch.cat(X)

    def __getitem__(self, item):
        return {
            'src': self.src[item],
            'coords': self.coords[item],
            'tgt': self.tgt[item],
            'path': self.path[item]
        } if self.path else {
            'src': self.src[item],
            'coords': self.coords[item],
            'tgt': self.tgt[item]
        }

    def __len__(self):
        return len(self.src)

    def get_image_size(self):
        return self.src[0].size()[1:]

    def get_dataset_infos(self):
        return {"image_size": self.get_image_size(),
                "n_prev": self.n_prev,
                "n_next": self.n_next,
                "block_size": self.block_size
                }

    @classmethod
    def conf_to_folders(cls, confname):

        if confname in ["biggest", "dc1", "deathcicle1"]:
            return ["deathCircle/video1/"]
        elif confname in ["sec_biggest", "dc3", "deathcicle3"]:
            return ["deathCircle/video3/"]
        elif confname in ["third_biggest", "n1", "nexus1"]:
            return ["nexus/video1/"]
        elif confname in ["dc0"]:
            return ["deathCircle/video0/"]
        elif confname in ["top_3"]:
            return ["deathCircle/video1/", "deathCircle/video3/", "nexus/video1/"]
        elif confname in ["b0", "bookstore0"]:
            return ["bookstore/video0/"]
        elif confname == "every_biggest":
            folders = ["bookstore/video0/"]
            folders += ["coupa/video3/"]
            folders += ["deathCircle/video1/"]
            folders += ["gates/video3/"]
            folders += ["hyang/video4/"]
            folders += ["little/video3/"]
            folders += ["nexus/video2/"]
            folders += ["quad/video2/"]
            return folders
        elif confname == "gates_1":
            folders = ["gates/video1/"]
            return folders
        elif confname == "all":
            folders = [f"bookstore/video{k}/" for k in range(7)]
            folders += [f"coupa/video{k}/" for k in range(4)]
            folders += [f"deathCircle/video{k}/" for k in range(5)]
            folders += [f"gates/video{k}/" for k in range(9)]
            folders += [f"hyang/video{k}/" for k in range(15)]
            folders += [f"little/video{k}/" for k in range(4)]
            folders += [f"nexus/video{k}/" for k in range(12)]
            folders += [f"quad/video{k}/" for k in range(4)]
            return folders
        elif confname == "DCS":
            folders = [f"deathCircle/video{k}/" for k in range(4)]
            return folders
        elif confname == "all_except_0":
            folders = [f"bookstore/video{k}/" for k in range(1, 7)]
            folders += [f"coupa/video{k}/" for k in range(1, 4)]
            folders += [f"deathCircle/video{k}/" for k in range(1, 5)]
            folders += [f"gates/video{k}/" for k in range(2, 9)]
            folders += [f"hyang/video{k}/" for k in range(1, 15)]
            folders += [f"little/video{k}/" for k in range(1, 4)]
            folders += [f"nexus/video{k}/" for k in [k for k in range(12) if k in [5, 6]]]
            folders += [f"quad/video{k}/" for k in range(1, 4)]
            return folders

        else:
            raise Exception("Dataset config name not recognized")
