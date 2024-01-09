import torch
import torch.nn as nn
import torch.nn.functional as F

class test(nn.Module):
    def __init__(self, cfg, prefix_len=10, emb_size=768): 
        super(test, self).__init__()

        # hyp        
        self.prefix_len = prefix_len 
        self.emb_size = emb_size
        self.hidden_size = cfg.SHALLOW.HIDDEN_SIZE
        self.special_token_size = cfg.SHALLOW.SPECIAL_TOKEN_SIZE
        
        # prompt 
        self.special_prompt = nn.Parameter(torch.empty(self.special_token_size, self.hidden_size))
        self.prompt = nn.Parameter(torch.empty(self.special_token_size, self.emb_size))
        
        # layers
        self.decoder1 = nn.Linear(self.hidden_size, 10)
        self.decoder2 = nn.Linear(10, self.special_token_size)
        
    
    def init_prompt(self):
        random_range = 0.5 
        self.prompt.data.uniform_(-random_range, random_range)
        self.special_prompt.data.uniform_(-random_range, random_range)
        
    
    def generate(self):
        
        special_token = self.decoder1(self.special_prompt)
        special_token = self.decoder2(special_token)
        
        x = special_token @ self.prompt       
        
        return x