import logging

from saicinpainting.training.modules.ffc import FFCResNetGenerator
from saicinpainting.training.modules.pix2pixhd import GlobalGenerator, MultiDilatedGlobalGenerator, \
    NLayerDiscriminator, MultidilatedNLayerDiscriminator

import pdb

def make_generator(config, kind, **kwargs):
    logging.info(f'Make generator {kind}')

    if kind == 'pix2pixhd_multidilated':
        return MultiDilatedGlobalGenerator(**kwargs)
    
    if kind == 'pix2pixhd_global':
        return GlobalGenerator(**kwargs)

    if kind == 'ffc_resnet':
        # kwargs = {
        #     'init_conv_kwargs': {
        #         'ratio_gin': 0,
        #         'ratio_gout': 0,
        #         'enable_lfu': False}, 
        #     'downsample_conv_kwargs': {
        #         'ratio_gin': '${generator.init_conv_kwargs.ratio_gout}', 
        #         'ratio_gout': '${generator.downsample_conv_kwargs.ratio_gin}', 
        #         'enable_lfu': False},
        #     'resnet_conv_kwargs': {
        #         'ratio_gin': 0.75, 
        #         'ratio_gout': '${generator.resnet_conv_kwargs.ratio_gin}', 
        #         'enable_lfu': False}
        # }
        # return FFCResNetGenerator(**kwargs)
        return FFCResNetGenerator()

    raise ValueError(f'Unknown generator kind {kind}')


def make_discriminator(kind, **kwargs):
    logging.info(f'Make discriminator {kind}')

    if kind == 'pix2pixhd_nlayer_multidilated':
        return MultidilatedNLayerDiscriminator(**kwargs)

    if kind == 'pix2pixhd_nlayer':
        return NLayerDiscriminator(**kwargs)

    raise ValueError(f'Unknown discriminator kind {kind}')
