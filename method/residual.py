import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    """ Residual Prompt Tuning: Improving Prompt Tuning with Residual Reparameterization (ACL 2023)"""
    def __init__(self, cfg, prefix_len=10, emb_size=768): 
        super(Residual, self).__init__()

        # hyp
        self.hidden_size = cfg.RESIDUAL.HIDDEN_SIZE
        self.drop_rate = cfg.RESIDUAL.DROP_RATE
        self.init_m = cfg.RESIDUAL.INIT_METHOD
        self.prefix_len = prefix_len 
        self.emb_size = emb_size

        # layers
        self.down_projector = nn.Linear(self.emb_size, self.hidden_size) # 10, 768
        self.up_projector = nn.Linear(self.hidden_size, self.emb_size)

        self.relu = nn.ReLU()
        self.layernorm = nn.LayerNorm(self.emb_size)
        self.dropout = nn.Dropout(self.drop_rate)

        # prompt 
        self.prompt = nn.Parameter(torch.empty(self.prefix_len, self.emb_size))

    def init_prompt(self):
        random_range = 0.5 
        self.prompt.data.uniform_(-random_range, random_range)
        
    
    def generate(self):
        x = self.prompt
        
        x = self.down_projector(x)
        x = self.relu(x)
        x = self.up_projector(x)
        x = self.layernorm(x)

        return x + self.prompt
        