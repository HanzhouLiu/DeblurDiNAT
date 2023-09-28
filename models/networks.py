import torch
import torch.nn as nn
from models import Stripformer, FSformer, FSformer_v0, FSformer_v1

def get_generator(model_config):
    generator_name = model_config['g_name']
    if generator_name == 'Stripformer':
        model_g = Stripformer.Stripformer()
    elif generator_name == 'FSformer':
        model_g = FSformer.FSformer()
    elif generator_name == 'FSformer_v0':
        model_g = FSformer_v0.FSformer_v0()
    elif generator_name == 'FSformer_v1':
        model_g = FSformer_v1.FSformer_v1()
    else:
        raise ValueError("Generator Network [%s] not recognized." % generator_name)
    return nn.DataParallel(model_g)

def get_nets(model_config):
    return get_generator(model_config)
