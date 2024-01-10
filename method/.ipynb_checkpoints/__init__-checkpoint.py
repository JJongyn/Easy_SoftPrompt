from .vanilla import Vanilla
from .residual import Residual
from .dpt import DPT

soft_prompt_dict = {
    "none": Vanilla,
    "residual": Residual,
    "dpt" : DPT,

}