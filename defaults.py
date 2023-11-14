# This file stores the default parameters of the experiment.
from yacs.config import CfgNode as CN

_C = CN()

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

_C.gpu = 0
_C.is_training = 1
_C.dataset = ""
_C.model = ""
_C.seq_len = 96
_C.label_len = 48
_C.pred_len = 96
_C.des = 'Exp'
_C.itr = 1








"""
# save the cfg as .yaml file
cfg = get_cfg_defaults()
with open("output.yaml", "w") as f:
  f.write(cfg.dump())   # save config to file
"""
