import torch
import torch.nn as nn

class BASE(nn.Module):
    def __init__(self):
        super(BASE, self).__init__()


    def init_prompt(self, x, y):
        random_range = 0.5
        x.data.uniform_(-random_range, random_range)
        y.data.uniform_(-random_range, random_range)