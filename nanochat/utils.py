import numpy as np
import torch
import os,random


class Config:
    def __init__(self, p):
        if isinstance(p,dict):
            dictionary=p
            self.__dict__=dictionary
        elif isinstance(p,str):
            import yaml
            with open(p,"r",encoding="utf-8") as file:
                dictionary=dict(yaml.safe_load(file))
            self.__dict__=dictionary



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False