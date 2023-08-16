import torch
from torch.optim.lr_scheduler import _LRScheduler

class NoamLR(_LRScheduler):
    def __init__(self, optimizer, model_dimension, warmup_steps,last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.model_dimension = model_dimension
        self.counter = 0
        super().__init__(optimizer)
        

    def get_lr(self):
        self.counter += 1 
        return [(self.model_dimension ** -0.5 * min(self.counter ** -0.5, self.counter * self.warmup_steps ** -1.5))/20]

