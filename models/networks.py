import torch
import torch.nn as nn
from models.DeblurDiNATL import NADeblurL
from models.DeblurDiNAT import NADeblur
def get_generator(model_config):
    generator_name = model_config['g_name']
    if generator_name == 'DeblurDiNAT':
        model_g = NADeblur()
    elif generator_name == 'DeblurDiNATL':
        model_g = NADeblurL()
    else:
        raise ValueError("Generator Network [%s] not recognized." % generator_name)
    return nn.DataParallel(model_g)

def get_nets(model_config):
    return get_generator(model_config)
