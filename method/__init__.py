from .vanilla import Vanilla
from .residual import Residual
from .dpt import DPT
from .prefix_tuning import Prefix_tuning

soft_prompt_dict = {
    "none": Vanilla,
    "vanilla": Vanilla,
    "prefix_tuning": Prefix_tuning,
    "residual": Residual,
    "dpt" : DPT,

}