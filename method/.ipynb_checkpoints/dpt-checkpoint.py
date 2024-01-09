import torch
import torch.nn as nn
import torch.nn.functional as F

class DPT(nn.Module):
    """ Decomposed Prompt Tuning via Low-Rank Reparameterization (EMNLP 2023)"""
    def __init__(self, cfg, prefix_len=10, emb_size=768): 
        super(DPT, self).__init__()

        # hyp
        self.hidden_size = cfg.DPT.HIDDEN_SIZE
        
        self.prefix_len = prefix_len 
        self.emb_size = emb_size

        
        # prompt 
        self.prompt_first = nn.Parameter(torch.empty(self.prefix_len, self.hidden_size))
        self.prompt_second = nn.Parameter(torch.empty(self.hidden_size, self.emb_size))

    def init_prompt(self):
        random_range = 0.5 
        self.prompt_first.data.uniform_(-random_range, random_range)
        self.prompt_second.data.uniform_(-random_range, random_range)

    
    def generate(self):
        return torch.matmul(self.prompt_first, self.prompt_second) 