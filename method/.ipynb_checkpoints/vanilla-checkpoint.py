import torch
import torch.nn as nn
import torch.nn.functional as F

class Vanilla(nn.Module):
    def __init__(self, cfg, prefix_len=10, emb_size=768): 
        super(Vanilla, self).__init__()

        # hyp        
        self.prefix_len = prefix_len 
        self.emb_size = emb_size

        # prompt 
        self.prompt = nn.Parameter(torch.empty(self.prefix_len, self.emb_size))

    def init_prompt(self):
        random_range = 0.5 
        self.prompt.data.uniform_(-random_range, random_range)
    
    def generate(self):
        return self.prompt