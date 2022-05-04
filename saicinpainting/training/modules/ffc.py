# Fast Fourier Convolution NeurIPS 2020
# original implementation https://github.com/pkumivision/FFC/blob/main/model_zoo/ffc.py
# paper https://proceedings.neurips.cc/paper/2020/file/2fd5d41ec6cfab47e32164d5624269b1-Paper.pdf

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from saicinpainting.training.modules.base import get_activation, BaseDiscriminator
# from saicinpainting.training.modules.spatial_transform import LearnableSpatialTransformWrapper
# from saicinpainting.training.modules.squeeze_excitation import SELayer
from saicinpainting.utils import get_shape

from typing import List
from typing import Tuple
from typing import Optional

import pdb

class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, 
                 ffc3d=False, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups

        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2,
                                          out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]

        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])


        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        return output


class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, **fu_kwargs):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        # fu_kwargs -- {}

        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)
        output = self.conv2(x + output)

        return output


class INIT_FFC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin = 0.0, ratio_gout = 0.0, stride=1, padding=0,
                 dilation=1, groups=1, bias=False,
                 padding_type='reflect'):
        super(INIT_FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride
        self.ratio_gin = ratio_gin # ==> 0.0
        self.ratio_gout = ratio_gout # ==> 0.0
        self.convl2l = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)

    def forward(self, x) -> Tuple[torch.Tensor, int]:
        out_xl = self.convl2l(x)
        return (out_xl, 0)


class INIT_FFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin = 0.0, ratio_gout = 0.0,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect'):
        super(INIT_FFC_BN_ACT, self).__init__()
        # INIT_FFC_BN_ACT (0.0, 0.0)

        self.ffc = INIT_FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, padding_type=padding_type)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        self.bn_l = lnorm(out_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        self.act_l = lact(inplace=True)

    def forward(self, x) -> Tuple[torch.Tensor, int]:
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))

        return (x_l, x_g)


class DOWN_FFC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin=0.0, ratio_gout=0.75, stride=1, padding=0,
                 dilation=1, groups=1, bias=False,
                 padding_type='reflect'):
        super(DOWN_FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride
        out_channels = out_channels - int(out_channels * ratio_gout)

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout

        self.convl2l = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)

    def forward(self, x: Tuple[torch.Tensor, int]) -> Tuple[torch.Tensor, int]:
        out_xl = self.convl2l(x[0])
        return (out_xl, 0)


class DOWN_FFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin = 0.0, ratio_gout = 0.75,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect'):
        super(DOWN_FFC_BN_ACT, self).__init__()
        # DOWN_FFC_BN_ACT (0.0, 0.75)

        self.ffc = DOWN_FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, padding_type=padding_type)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        # gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        lact = nn.Identity if ratio_gout == 1 else activation_layer
        self.act_l = lact(inplace=True)

    def forward(self, x: Tuple[torch.Tensor, int]) -> Tuple[torch.Tensor, int]:
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))

        return (x_l, x_g)


class START_REST_FFC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin = 0.75, ratio_gout = 0.75, stride=1, padding=0,
                 dilation=1, groups=1, bias=False,
                 padding_type='reflect'):
        super(START_REST_FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        #groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        #groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        # print(" ###### REST_FFC self.ratio_gin, self.ratio_gout --", self.ratio_gin, self.ratio_gout)
        # print(" ###### REST_FFC in_cl, out_cl --", in_cl, out_cl)
        # print(" ###### REST_FFC in_cg, out_cg --", in_cg, out_cg)

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)

        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)

        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2)


    def forward(self, x: Tuple[torch.Tensor, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        # xxxx8888
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = x_l, 0

        g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1: # True
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g)

        # print("--- REST_FFC type(x_g)", type(x), type(x_g))

        return (out_xl, out_xg)


class START_REST_FFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin = 0.75, ratio_gout = 0.75,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect'):
        super(START_REST_FFC_BN_ACT, self).__init__()
        # kargs -- {}

        self.ffc = START_REST_FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, padding_type=padding_type)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x: Tuple[torch.Tensor, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        # xxxx8888
        print("--- START_REST_FFC_BN_ACT { type(x): ", type(x))
        if isinstance(x, tuple):
            print("  --- x type: ", type(x[0]), type(x[1]))
        else:
            print("  --- x type: ", type(x))

        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))

        print("  --- type(x_g): ", type(x_g), "}")

        return (x_l, x_g)


class REST_FFC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin=0.75, ratio_gout=0.75, stride=1, padding=0,
                 dilation=1, groups=1, bias=False,
                 padding_type='reflect'):
        super(REST_FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        #groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        #groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        # print(" ###### REST_FFC self.ratio_gin, self.ratio_gout --", self.ratio_gin, self.ratio_gout)
        # print(" ###### REST_FFC in_cl, out_cl --", in_cl, out_cl)
        # print(" ###### REST_FFC in_cg, out_cg --", in_cg, out_cg)

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)

        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)

        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2)


    def forward(self, x: Tuple[torch.Tensor, int]) -> Tuple[torch.Tensor, int]:
        # xxxx8888
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = x_l, 0

        g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1: # True
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g)

        # print("--- REST_FFC type(x_g)", type(x), type(x_g))

        return out_xl, out_xg

class REST_FFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin=0.75, ratio_gout=0.75,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect'):
        super(REST_FFC_BN_ACT, self).__init__()
        # kargs -- {}

        self.ffc = REST_FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, padding_type=padding_type)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # xxxx8888
        print("--- REST_FFC_BN_ACT { type(x): ", type(x))
        if isinstance(x, tuple):
            print("  --- x type: ", type(x[0]), type(x[1]))
        else:
            print("  --- x type: ", type(x))

        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))

        print("  --- type(x_g): ", type(x_g), "}")

        return (x_l, x_g)


class FFCResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
                 spatial_transform_kwargs=None, inline=False, **conv_kwargs):
        super().__init__()
        # conv_kwargs:  {'ratio_gin': 0.75, 'ratio_gout': 0.75}
        self.conv1 = REST_FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type)
        self.conv2 = REST_FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type)
        # if spatial_transform_kwargs is not None: # False
        #     self.conv1 = LearnableSpatialTransformWrapper(self.conv1, **spatial_transform_kwargs)
        #     self.conv2 = LearnableSpatialTransformWrapper(self.conv2, **spatial_transform_kwargs)
        self.inline = inline # False

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        # xxxx8888 ==> root ?

        # xxxx8888
        # x_l, x_g = x if type(x) is tuple else (x, 0)
        x_l, x_g = x

        id_l, id_g = x_l, x_g

        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))

        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g

        # if self.inline:
        #     out = torch.cat(out, dim=1)

        return out


class ReduceTupleLayer(nn.Module):
    def forward(self, x: Tuple[torch.Tensor, int]):
        # assert isinstance(x, tuple)
        x_l, x_g = x
        # assert torch.is_tensor(x_l) or torch.is_tensor(x_g)
        # if not torch.is_tensor(x_g):
        #     print(" --- ReduceTupleLayer forward x_g: ", type(x_g), x_g)
        #     return x_l
        return torch.cat(x, dim=1)


class FFCResNetGenerator(nn.Module):
    def __init__(self, input_nc=4, output_nc=3, ngf=64, n_downsampling=3, n_blocks=18, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', activation_layer=nn.ReLU,
                 up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                 # init_conv_kwargs={}, downsample_conv_kwargs={}, resnet_conv_kwargs={},
                 spatial_transform_layers=None, spatial_transform_kwargs={},
                 add_out_act='sigmoid', max_features=1024, out_ffc=False, out_ffc_kwargs={}):
        assert (n_blocks >= 0)
        super().__init__()

        init_conv_kwargs = {
            'ratio_gin': 0.0,
            'ratio_gout': 0.0,
        }
        resnet_conv_kwargs = {
            'ratio_gin': 0.75,
            'ratio_gout': 0.75,
        }
        #     'downsample_conv_kwargs': {'ratio_gin': '${generator.init_conv_kwargs.ratio_gout}', 
        #         'ratio_gout': '${generator.downsample_conv_kwargs.ratio_gin}'},
        downsample_conv_kwargs = {
            'ratio_gin': init_conv_kwargs['ratio_gout'], 
            'ratio_gout':  init_conv_kwargs['ratio_gin'], # resnet_conv_kwargs['ratio_gin']
        }

        # INIT_FFC_BN_ACT (0.0, 0.0)
        # DOWN_FFC_BN_ACT (0.0, 0.75)
        # REST_FFC_BN_ACT (0.75, 0.75)


        model = [nn.ReflectionPad2d(3),
                 INIT_FFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0, norm_layer=norm_layer,
                            activation_layer=activation_layer, **init_conv_kwargs)]

        ### downsample
        for i in range(n_downsampling): # n_downsampling -- 3
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                # resnet_conv_kwargs.get('ratio_gin', 0) -- 0.75
                cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs

            if i == n_downsampling - 1:
                model += [START_REST_FFC_BN_ACT(min(max_features, ngf * mult),
                                 min(max_features, ngf * mult * 2),
                                 kernel_size=3, stride=2, padding=1,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer,
                                 **cur_conv_kwargs)]
            else:
                model += [DOWN_FFC_BN_ACT(min(max_features, ngf * mult),
                                 min(max_features, ngf * mult * 2),
                                 kernel_size=3, stride=2, padding=1,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer,
                                 **cur_conv_kwargs)]


        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)
        ### resnet blocks
        for i in range(n_blocks): # n_blocks -- 18
            cur_resblock = FFCResnetBlock(feats_num_bottleneck, padding_type=padding_type, activation_layer=activation_layer,
                                          norm_layer=norm_layer, **resnet_conv_kwargs)
            # spatial_transform_layers -- None 
            # if spatial_transform_layers is not None and i in spatial_transform_layers:
            #     cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
            model += [cur_resblock]

        model += [ReduceTupleLayer()]

        ### upsample
        for i in range(n_downsampling): # n_downsampling -- 3
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(max_features, ngf * mult),
                                         min(max_features, int(ngf * mult / 2)),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))),
                      up_activation]

        if out_ffc: # False
            model += [FFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer,
                                     norm_layer=norm_layer, inline=True, **out_ffc_kwargs)]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act: # add_out_act -- 'sigmoid'
            model.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.model = nn.Sequential(*model)

    def forward(self, input):
        # input.size() -- [1, 4, 1000, 1504], input[:, 3:4, :, :].mean() -- 0.1590
        output = self.model(input)
        # output.size() -- [1, 3, 1000, 1504]
        return output


# class FFCNLayerDiscriminator(BaseDiscriminator):
#     def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, max_features=512,
#                  init_conv_kwargs={}, conv_kwargs={}):
#         super().__init__()
#         self.n_layers = n_layers

#         def _act_ctor(inplace=True):
#             return nn.LeakyReLU(negative_slope=0.2, inplace=inplace)

#         kw = 3
#         padw = int(np.ceil((kw-1.0)/2))
#         sequence = [[FFC_BN_ACT(input_nc, ndf, kernel_size=kw, padding=padw, norm_layer=norm_layer,
#                                 activation_layer=_act_ctor, **init_conv_kwargs)]]

#         nf = ndf
#         for n in range(1, n_layers):
#             nf_prev = nf
#             nf = min(nf * 2, max_features)

#             cur_model = [
#                 FFC_BN_ACT(nf_prev, nf,
#                            kernel_size=kw, stride=2, padding=padw,
#                            norm_layer=norm_layer,
#                            activation_layer=_act_ctor,
#                            **conv_kwargs)
#             ]
#             sequence.append(cur_model)

#         nf_prev = nf
#         nf = min(nf * 2, 512)

#         cur_model = [
#             FFC_BN_ACT(nf_prev, nf,
#                        kernel_size=kw, stride=1, padding=padw,
#                        norm_layer=norm_layer,
#                        activation_layer=lambda *args, **kwargs: nn.LeakyReLU(*args, negative_slope=0.2, **kwargs),
#                        **conv_kwargs),
#             ReduceTupleLayer()
#         ]
#         sequence.append(cur_model)

#         sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

#         for n in range(len(sequence)):
#             setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))

#     def get_all_activations(self, x):
#         res = [x]
#         for n in range(self.n_layers + 2):
#             model = getattr(self, 'model' + str(n))
#             res.append(model(res[-1]))
#         return res[1:]

#     def forward(self, x):
#         act = self.get_all_activations(x)
#         feats = []
#         for out in act[:-1]:
#             if isinstance(out, tuple):
#                 if torch.is_tensor(out[1]):
#                     out = torch.cat(out, dim=1)
#                 else:
#                     out = out[0]
#             feats.append(out)
#         return act[-1], feats


if __name__ == '__main__':
    model = FFCResNetGenerator()
    model = torch.jit.script(model)

    pdb.set_trace()
