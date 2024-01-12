import torch
import torch.nn as nn
import torch.nn.functional as F


class Prefix_tuning(nn.Module):
    """ Prefix-Tuning: Optimizing Continuous Prompts for Generation (ACL 2021) """
    def __init__(self, cfg, prefix_len=10, emb_size=768): 
        super(Prefix_tuning, self).__init__()

        # hyp
        self.hidden_size = cfg.PREFIX.HIDDEN_SIZE
        self.prefix_len = prefix_len 
        self.emb_size = emb_size

        # layers
        self.down_projector = nn.Linear(self.emb_size, self.hidden_size) # 768, 512
        self.up_projector = nn.Linear(self.hidden_size, self.emb_size)

        if cfg.PREFIX.ACTIVATION == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()
            
        # prompt 
        self.prompt = nn.Parameter(torch.empty(self.prefix_len, self.emb_size))

    def init_prompt(self):
        random_range = 0.5 
        self.prompt.data.uniform_(-random_range, random_range)
        
    
    def generate(self):
        x = self.prompt
        
        x = self.down_projector(x)
        x = self.activation(x)
        x = self.up_projector(x)

        return x 
        