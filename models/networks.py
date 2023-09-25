import torch
import torch.nn as nn
from models.Stripformer import Stripformer
from models.FSformer import FSformer

def get_generator(model_config):
    generator_name = model_config['g_name']
    if generator_name == 'Stripformer':
        model_g = Stripformer()
    elif generator_name == 'FSformer':
        model_g = FSformer()
    else:
        raise ValueError("Generator Network [%s] not recognized." % generator_name)
    return nn.DataParallel(model_g)

def get_nets(model_config):
    return get_generator(model_config)
