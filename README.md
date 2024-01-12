# Easy Soft Prompt
This repository contains an implementation of Soft Prompt tuning, a parameter-efficient fine-tuning method for LLM. The methods included are listed below.

- Prefix-Tuning: Optimizing Continuous Prompts for Generation (ACL 2021)
- Residual Prompt Tuning: Improving Prompt Tuning with Residual Reparameterization (ACL Findings 2023)
- Decomposed Prompt Tuning via Low-Rank Reparameterization (EMNLP Findings 2023)

## Usage
We currently only provide simple functionality for soft prompt tuning. 
#### 1. Add your prompt configure to cfg.py
    # RESIDUAL
    CFG.RESIDUAL = CN()
    CFG.RESIDUAL.HIDDEN_SIZE = 400
    CFG.RESIDUAL.DROP_RATE = 0.05
    CFG.RESIDUAL.INIT_METHOD = 'random'

    # Add your prompt settings
    CFG."***" = CN()
    CFG."***".HIDDEN_SIZE = 10

#### 2. Create your soft prompt
You can create a prompt as shown in the example below. Note that you need to register your methods in *init.py*
```python
class Vanilla(nn.Module):
"""The Power of Scale for Parameter-Efficient Prompt Tuning (EMNLP 2021)"""
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
```
#### 3. Training
```python
CUDA_VISIBLE_DEVICES=0 python main.py --datasets='copa' --model_name=vanilla --enc_prompt_tokens 10 -ts 16 -e 10 --save_name Vanilla
```


---
### Our code is based on
- https://github.com/XYaoooo/DPT
- https://github.com/arazd/ResidualPrompts
