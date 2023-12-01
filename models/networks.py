import torch
import torch.nn as nn
from models import Stripformer, NADeblur_V0, NADeblur_V1, NADeblur_V2, NADeblur_V3, NADeblur_V4, NADeblur_V5, NADeblur_V6, NADeblur_V7, NADeblur_V8, NADeblur_V9, NADeblur_V10, NADeblur_V11

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
    elif generator_name == 'NADeblur_V4':
        model_g = NADeblur_V4.NADeblur_V4()
    elif generator_name == 'NADeblur_V5':
        model_g = NADeblur_V5.NADeblur_V5()
    elif generator_name == 'NADeblur_V6':
        model_g = NADeblur_V6.NADeblur_V6()
    elif generator_name == 'NADeblur_V7':
        model_g = NADeblur_V7.NADeblur_V7()
    elif generator_name == 'NADeblur_V8':
        model_g = NADeblur_V8.NADeblur_V8()
    elif generator_name == 'NADeblur_V9':
        model_g = NADeblur_V9.NADeblur_V9()
    elif generator_name == 'NADeblur_V10':
        model_g = NADeblur_V10.NADeblur_V10()
    elif generator_name == 'NADeblur_V11':
        model_g = NADeblur_V11.NADeblur_V11()
    else:
        raise ValueError("Generator Network [%s] not recognized." % generator_name)
    return nn.DataParallel(model_g)

def get_nets(model_config):
    return get_generator(model_config)
