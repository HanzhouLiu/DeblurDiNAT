import torch
import torch.nn as nn
from models import Stripformer, FSformer, FSformer_v0, FSformer_v1, FSformer_B, FSformer_B1, FSformer_B2, FSformer_V2, FSformer_V3, FSformer_V4, FSformer_V5, FSformer_V6, FSformer_V7, FSformer_V8

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
    elif generator_name == 'FSformer_B':
        model_g = FSformer_B.FSformer_B()
    elif generator_name == 'FSformer_B1':
        model_g = FSformer_B1.FSformer_B1()
    elif generator_name == 'FSformer_B2':
        model_g = FSformer_B2.FSformer_B2()
    elif generator_name == 'FSformer_V2':
        model_g = FSformer_V2.FSformer_V2()
    elif generator_name == 'FSformer_V3':
        model_g = FSformer_V3.FSformer_V3()
    elif generator_name == 'FSformer_V4':
        model_g = FSformer_V4.FSformer_V4()
    elif generator_name == 'FSformer_V5':
        model_g = FSformer_V5.FSformer_V5()
    elif generator_name == 'FSformer_V6':
        model_g = FSformer_V6.FSformer_V6()
    elif generator_name == 'FSformer_V7':
        model_g = FSformer_V7.FSformer_V7()
    elif generator_name == 'FSformer_V8':
        model_g = FSformer_V8.FSformer_V8()
    else:
        raise ValueError("Generator Network [%s] not recognized." % generator_name)
    return nn.DataParallel(model_g)

def get_nets(model_config):
    return get_generator(model_config)
