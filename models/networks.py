import torch
import torch.nn as nn
from models import Stripformer, NADeblur_V0, NADeblur_V1, NADeblur_V2, NADeblur_V3

def get_generator(model_config):
    generator_name = model_config['g_name']
    if generator_name == 'Stripformer':
        model_g = Stripformer.Stripformer()
    elif generator_name == 'NADeblur_V0':
        model_g = NADeblur_V0.NADeblur_V0()
    elif generator_name == 'NADeblur_V1':
        model_g = NADeblur_V1.NADeblur_V1()
    elif generator_name == 'NADeblur_V2':
        model_g = NADeblur_V2.NADeblur_V2()
    elif generator_name == 'NADeblur_V3':
        model_g = NADeblur_V3.NADeblur_V3()
    else:
        raise ValueError("Generator Network [%s] not recognized." % generator_name)
    return nn.DataParallel(model_g)

def get_nets(model_config):
    return get_generator(model_config)
