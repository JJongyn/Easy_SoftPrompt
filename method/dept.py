import torch
import torch.nn as nn
import torch.nn.functional as F

import math 

class DePT(nn.Module):
    """ DePT: Decomposed Prompt Tuning for Parameter-Efficient Fine-tuning (ICLR2023 underReview) """
    def __init__(self, cfg, prefix_len=10, emb_size=768): 
        super(DePT, self).__init__()

        # hyp
        self.r = cfg.DEPT.R
        self.alpha = cfg.DEPT.ALPHA
        self.scaling = self.alpha / math.sqrt(self.r)
        
        self.prefix_len = prefix_len 
        self.emb_size = emb_size

        # prompt 
        self.lora_embedding_A  = nn.Parameter(torch.empty(self.prefix_len, self.r))
        self.lora_embedding_B  = nn.Parameter(torch.empty(self.r, self.emb_size))

    def init_prompt(self):
        random_range = 0.5 
        self.lora_embedding_A.data.uniform_(-random_range, random_range)
        self.lora_embedding_B.data.uniform_(-random_range, random_range)

    
    def generate(self):
        return self.scaling * (self.lora_embedding_A @ self.lora_embedding_B)