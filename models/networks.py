import torch
import torch.nn as nn
from models import Stripformer, NADeblur_V0, NADeblur_V1, NADeblur_V2, NADeblur_V3, NADeblur_V4, NADeblur_V5, NADeblur_V6, NADeblur_V7, NADeblur_V8, NADeblur_V9, NADeblur_V10, NADeblur_V11, NADeblur_V12, NADeblur_V13, NADeblur_V14, NADeblur_V15, NADeblur_V16, NADeblur_V17, NADeblur_V18, NADeblur_V19, NADeblur_V20, NADeblur_V21, NADeblur_V22, NADeblur_V23, NADeblur_V24, NADeblur
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
    elif generator_name == 'NADeblur_V12':
        model_g = NADeblur_V12.NADeblur_V12()
    elif generator_name == 'NADeblur_V13':
        model_g = NADeblur_V13.NADeblur_V13()
    elif generator_name == 'NADeblur_V14':
        model_g = NADeblur_V14.NADeblur_V14()
    elif generator_name == 'NADeblur_V15':
        model_g = NADeblur_V15.NADeblur_V15()
    elif generator_name == 'NADeblur_V16':
        model_g = NADeblur_V16.NADeblur_V16()
    elif generator_name == 'NADeblur_V17':
        model_g = NADeblur_V17.NADeblur_V17()
    elif generator_name == 'NADeblur_V18':
        model_g = NADeblur_V18.NADeblur_V18()
    elif generator_name == 'NADeblur_V19':
        model_g = NADeblur_V19.NADeblur_V19()
    elif generator_name == 'NADeblur_V20':
        model_g = NADeblur_V20.NADeblur_V20()
    elif generator_name == 'NADeblur_V21':
        model_g = NADeblur_V21.NADeblur_V21()
    elif generator_name == 'NADeblur_V22':
        model_g = NADeblur_V22.NADeblur_V22()
    elif generator_name == 'NADeblur_V23':
        model_g = NADeblur_V23.NADeblur_V23()
    elif generator_name == 'NADeblur_V24':
        model_g = NADeblur_V24.NADeblur_V24()
    elif generator_name == 'NADeblur':
        model_g = NADeblur.NADeblur()
    else:
        raise ValueError("Generator Network [%s] not recognized." % generator_name)
    return nn.DataParallel(model_g)

def get_nets(model_config):
    return get_generator(model_config)
