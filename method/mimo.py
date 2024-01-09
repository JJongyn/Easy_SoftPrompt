import torch
import torch.nn as nn
import torch.nn.functional as F

class mimo(nn.Module):
    def __init__(self, cfg, prefix_len=10, emb_size=768): 
        super(mimo, self).__init__()

        # hyp        
        self.prefix_len = prefix_len 
        self.emb_size = emb_size

        # prompt 
        self.prompt1 = nn.Parameter(torch.empty(self.prefix_len // 2, self.emb_size))
        self.prompt2 = nn.Parameter(torch.empty(self.prefix_len // 2, self.emb_size))

        # layers
        self.input1 = nn.Linear(self.emb_size, 50)
        self.input2 = nn.Linear(self.emb_size, 50)

        self.shared_layer = nn.Linear(50, 50)

        self.output1 = nn.Linear(50, self.emb_size)
        self.output2 = nn.Linear(50, self.emb_size)
    
    def init_prompt(self):
        random_range = 0.5 
        self.prompt1.data.uniform_(-random_range, random_range)
        self.prompt2.data.uniform_(-random_range, random_range)
    
    def generate(self):
        
        input1 = self.input1(self.prompt1)
        input2 = self.input2(self.prompt2)

        input1 = self.shared_layer(input1)
        input2 = self.shared_layer(input2)

        input1 = self.output1(input1)
        input2 = self.output2(input2)

        x = torch.concat([input1, input2], dim = 0)
        
        return x