from .vanilla import Vanilla
from .residual import Residual
from .dpt import DPT
from .mimo import mimo
from .shallow_dpt import shallow_dpt
from .test import test

soft_prompt_dict = {
    "none": Vanilla,
    "residual": Residual,
    "dpt" : DPT,
    "mimo" : mimo,
    "shallow_dpt": shallow_dpt,
    "test": test,

}