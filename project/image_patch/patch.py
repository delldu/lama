# Fast Fourier Convolution NeurIPS 2020
# original implementation https://github.com/pkumivision/FFC/blob/main/model_zoo/ffc.py
# paper https://proceedings.neurips.cc/paper/2020/file/2fd5d41ec6cfab47e32164d5624269b1-Paper.pdf

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function
from torch.onnx import symbolic_helper

from typing import Tuple

import todos
import pdb

# https://zhuanlan.zhihu.com/p/653745531
# https://github.com/onnx/onnx/issues/4845
# https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.Rfft
# https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.Irfft
# https://github.com/Alexey-Kamenev/tensorrt-dft-plugins
# https://github.com/microsoft/onnxruntime/issues/13279

# torch.fft.rfft2 is same as rfftn for 4D-tensor
# RFFT is only supported on GPUs from https://github.com/NVIDIA/modulus/issues/42
class OnnxRfft2(Function):
    @staticmethod
    def forward(ctx, input) -> torch.Value:
        # todos.debug.output_var("rfft2.input", input)
        y1 =  torch.fft.rfft2(input, dim=(-2, -1), norm="backward")
        # todos.debug.output_var("rfft2.y1", y1)

        y2 =  torch.view_as_real(y1)
        # todos.debug.output_var("rfft2.y2", y2)

        return y2

    @staticmethod
    def symbolic(g: torch.Graph, input: torch.Value) -> torch.Value:
        return g.op(
            "com.microsoft::Rfft",
            input, normalized_i=0, onesided_i=1, signal_ndim_i=2
        )
onnx_rfftn = OnnxRfft2.apply


class OnnxComplex(Function):
    """Auto-grad function to mimic irfft for ONNX exporting
    """
    @staticmethod
    def forward(ctx, input1, input2):
        # todos.debug.output_var("Complex.input1", input1)
        # todos.debug.output_var("Complex.input2", input2)

        y = torch.complex(input1, input2)
        # todos.debug.output_var("Complex.y", y)
        return y

    @staticmethod
    def symbolic(g: torch.Graph, input1: torch.Value, input2: torch.Value) -> torch.Value:
        """Symbolic representation for onnx graph"""
        input1 = symbolic_helper._unsqueeze_helper(g, input1, [-1])
        input2 = symbolic_helper._unsqueeze_helper(g, input2, [-1])
        return g.op("Concat", input1, input2, axis_i=-1)
onnx_complex = OnnxComplex.apply


# torch.fft.irfft2 is same as irfftn for 4D-tensor
class OnnxIrfft2(Function):
    @staticmethod
    def forward(ctx, input) -> torch.Value:
        # return torch.fft.irfft2(
        #     torch.view_as_complex(input), dim=(-2, -1), norm="backward"
        # )
        # input is Complex ..., size() -- [1, 192, 128, 65]
        # todos.debug.output_var("Irfft2.input", input)
        y = torch.fft.irfft2(input, dim=(-2, -1), norm="backward") # output is real, size() -- [1, 192, 128, 128]
        # todos.debug.output_var("Irfft2.y", y)
        return y

    @staticmethod
    def symbolic(g: torch.Graph, input: torch.Value) -> torch.Value:
        return g.op(
            "com.microsoft::Irfft", input, normalized_i=0, onesided_i=1, signal_ndim_i=2
            )
onnx_irfftn = OnnxIrfft2.apply


class FourierUnit(nn.Module):
    '''2D Fourier'''
    def __init__(self, in_channels, out_channels, groups=1):
        super().__init__()
        assert (in_channels == 192 and out_channels == 192 and groups == 1)

        self.groups = groups
        self.conv_layer = nn.Conv2d(
            in_channels=in_channels * 2,
            out_channels=out_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=self.groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels * 2)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        B, C, H, W = x.size()

        # x.size() -- [1, 192, 128, 128]
        # ffted = torch.fft.rfftn(x, dim=(2, 3), norm="ortho")
        ffted = onnx_rfftn(x)

        # ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = torch.stack((ffted[..., 0], ffted[..., 1]), dim=-1) # [1, 192, 128, 65, 2]

        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # [1, 192, 2, 128, 65], (B, c, 2, h, w/2+1)
        ffted = ffted.view((B, -1, ) + ffted.size()[3:]) # [1, 384, 128, 65]

        ffted = self.conv_layer(ffted)  # [1, 384, 128, 65] (B, c*2, h, w/2+1)
        # ffted = self.relu(self.bn(ffted))
        ffted = F.relu(self.bn(ffted))

        ffted = (
            ffted.view((B, -1, 2, ) + ffted.size()[2:])
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        ) # [1, 192, 128, 65, 2] torch.float32

        # ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        ffted = onnx_complex(ffted[..., 0], ffted[..., 1])
        # ffted.size() -- [1, 192, 128, 65], Complex

        # output = torch.fft.irfftn(ffted, dim=(2, 3), norm="ortho")
        output = onnx_irfftn(ffted)

        return output # [1, 192, 128, 128], real


class SpectralTransform(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super().__init__()
        assert (in_channels == 384 and out_channels == 384 and stride == 1 and groups == 1)

        if stride == 2:
            pdb.set_trace()
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)


    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)
        output = self.conv2(x + output)

        return output

    def extra_repr(self) -> str:
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}'


class INIT_FFC(nn.Module):
    """FFC -- Fast Fourier Convolution"""

    def __init__(self, in_channels, out_channels, kernel_size,
        ratio_gin=0.0,
        ratio_gout=0.0,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super().__init__()
        assert ratio_gin == 0.0 and ratio_gout == 0.0

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.convl2l = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, 
            dilation, groups, bias, padding_mode="reflect"
        )

    def forward(self, x) -> Tuple[torch.Tensor, int]:
        out_xl = self.convl2l(x)
        return (out_xl, 0)

    def extra_repr(self) -> str:
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}'


class INIT_FFC_BN_ACT(nn.Module):
    """FFC -- Fast Fourier Convolution"""

    def __init__(self, in_channels, out_channels, kernel_size,
        ratio_gin=0.0,
        ratio_gout=0.0,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super().__init__()
        assert ratio_gin == 0.0 and ratio_gout == 0.0

        self.ffc = INIT_FFC(
            in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, stride, padding, 
            dilation, groups, bias
        )
        self.bn_l = nn.BatchNorm2d(out_channels)
        # self.act_l = nn.ReLU()

    def forward(self, x) -> Tuple[torch.Tensor, int]:
        x_l, x_g = self.ffc(x)
        # x_l = self.act_l(self.bn_l(x_l))
        x_l = F.relu(self.bn_l(x_l))

        return (x_l, x_g)

    def extra_repr(self) -> str:
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}'


class DOWN_FFC(nn.Module):
    """FFC -- Fast Fourier Convolution"""

    def __init__(self, in_channels, out_channels, kernel_size,
        ratio_gin = 0.0,
        ratio_gout = 0.0,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super().__init__()
        assert ratio_gin == 0.0 and ratio_gout == 0.0
        assert stride == 1 or stride == 2, "Stride should be 1 or 2."

        # out_channels = out_channels - int(out_channels * ratio_gout)
        self.convl2l = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, 
            dilation, groups, bias, padding_mode="reflect"
        )

    def forward(self, x: Tuple[torch.Tensor, int]) -> Tuple[torch.Tensor, int]:
        out_xl = self.convl2l(x[0])
        return (out_xl, 0)

    def extra_repr(self) -> str:
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}'


class DOWN_FFC_BN_ACT(nn.Module):
    """FFC -- Fast Fourier Convolution"""

    def __init__(self, in_channels, out_channels, kernel_size,
        ratio_gin=0.0,
        ratio_gout=0.0,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super().__init__()
        assert ratio_gin == 0.0 and ratio_gout == 0.0

        self.ffc = DOWN_FFC(
            in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, 
            stride, padding, dilation, groups, bias
        )
        # out_channels = out_channels - int(out_channels * ratio_gout) # 128 | 256

        self.bn_l = nn.BatchNorm2d(out_channels)
        # self.act_l = nn.ReLU()

    def forward(self, x: Tuple[torch.Tensor, int]) -> Tuple[torch.Tensor, int]:
        x_l, x_g = self.ffc(x)
        # x_l = self.act_l(self.bn_l(x_l))
        x_l = F.relu(self.bn_l(x_l))

        return (x_l, x_g)

    def extra_repr(self) -> str:
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}'


class START_REST_FFC(nn.Module):
    """FFC -- Fast Fourier Convolution"""

    def __init__(self, in_channels, out_channels, kernel_size,
        ratio_gin=0.75,
        ratio_gout=0.75,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super().__init__()
        assert ratio_gin == 0.0 and ratio_gout == 0.75
        assert stride == 1 or stride == 2, "Stride should be 1 or 2."

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        self.convl2l = nn.Conv2d(in_cl, out_cl, kernel_size, stride, padding, 
            dilation, groups, bias, padding_mode="reflect")
        self.convl2g = nn.Conv2d(in_cl, out_cg, kernel_size, stride, padding, 
            dilation, groups, bias, padding_mode="reflect")

    def forward(self, x: Tuple[torch.Tensor, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        x_l, x_g = x
        out_xl = self.convl2l(x_l)
        out_xg = self.convl2g(x_l)
        return (out_xl, out_xg)

    def extra_repr(self) -> str:
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}'


class START_REST_FFC_BN_ACT(nn.Module):
    """FFC -- Fast Fourier Convolution"""

    def __init__(self, in_channels, out_channels, kernel_size,
        ratio_gin=0.0,
        ratio_gout=0.75,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super().__init__()
        assert ratio_gin == 0.0 and ratio_gout == 0.75

        self.ffc = START_REST_FFC(
            in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, 
            stride, padding, dilation, groups, bias
        )
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = nn.BatchNorm2d(out_channels - global_channels)
        self.bn_g = nn.BatchNorm2d(global_channels)

    def forward(self, x: Tuple[torch.Tensor, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        x_l, x_g = self.ffc(x)
        x_l = F.relu(self.bn_l(x_l))
        x_g = F.relu(self.bn_g(x_g))

        return (x_l, x_g)

    def extra_repr(self) -> str:
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}'


class REST_FFC(nn.Module):
    """FFC -- Fast Fourier Convolution"""

    def __init__(self, in_channels, out_channels, kernel_size,
        ratio_gin=0.75,
        ratio_gout=0.75,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super().__init__()
        assert ratio_gin == 0.75 and ratio_gout == 0.75
        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        assert bias == False, "bias == False"

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        self.convl2l = nn.Conv2d(in_cl, out_cl, kernel_size, stride, padding, 
            dilation, groups, bias, padding_mode="reflect")
        self.convl2g = nn.Conv2d(in_cl, out_cg, kernel_size, stride, padding, 
            dilation, groups, bias, padding_mode="reflect")
        self.convg2l = nn.Conv2d(in_cg, out_cl, kernel_size, stride, padding, 
            dilation, groups, bias, padding_mode="reflect")
        self.convg2g = SpectralTransform(in_cg, out_cg, stride, 1 if groups == 1 else groups // 2)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x_l, x_g = x
        out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        out_xg = self.convl2g(x_l) + self.convg2g(x_g)
        return (out_xl, out_xg)

    def extra_repr(self) -> str:
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}'


class REST_FFC_BN_ACT(nn.Module):
    """FFC -- Fast Fourier Convolution"""

    def __init__(self, in_channels, out_channels, kernel_size,
        ratio_gin=0.75,
        ratio_gout=0.75,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super().__init__()
        # out_channels = 512
        # kernel_size = 3
        # ratio_gin = 0.75
        # ratio_gout = 0.75
        # stride = 1
        # padding = 1
        # dilation = 1
        # groups = 1
        # bias = False

        assert ratio_gin == 0.75 and ratio_gout == 0.75
        assert (bias == False)

        self.ffc = REST_FFC(
            in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, 
            stride, padding, dilation, groups, bias
        )

        global_channels = int(out_channels * ratio_gout) # 384
        self.bn_l = nn.BatchNorm2d(out_channels - global_channels) # 128
        self.bn_g = nn.BatchNorm2d(global_channels)
        # self.act_l = nn.ReLU()
        # self.act_g = nn.ReLU()

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x_l, x_g = self.ffc(x)
        # x_l = self.act_l(self.bn_l(x_l))
        # x_g = self.act_g(self.bn_g(x_g))

        x_l = F.relu(self.bn_l(x_l))
        x_g = F.relu(self.bn_g(x_g))

        return (x_l, x_g)

    def extra_repr(self) -> str:
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}'


class FFCResnetBlock(nn.Module):
    """FFC -- Fast Fourier Convolution"""

    def __init__(self, dim, dilation=1):
        super().__init__()
        self.dim = dim
        self.dilation = dilation
        assert dilation == 1

        self.conv1 = REST_FFC_BN_ACT(dim, dim,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.conv2 = REST_FFC_BN_ACT(dim, dim,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x_l, x_g = x
        id_l, id_g = x_l, x_g

        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))

        x_l, x_g = id_l + x_l, id_g + x_g

        return (x_l, x_g)

    def extra_repr(self) -> str:
        return f'dim={self.dim}, dialtion={self.dilation}'


class ReduceTupleLayer(nn.Module):
    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        # todos.debug.output_var("ReduceTupleLayer.x", x)

        return torch.cat(x, dim=1)


class FFCResNetGenerator(nn.Module):
    """FFC -- Fast Fourier Convolution"""
    def __init__(self,
        input_nc=4,
        output_nc=3,
        ngf=64,
        n_downsampling=3,
        n_blocks=18,
    ):
        super().__init__()
        self.MAX_H = 2048
        self.MAX_W = 4096
        self.MAX_TIMES = 32
        # GPU -- 2048x4096, 6.8G, 2120ms

        resnet_conv_kwargs = {
            "ratio_gin": 0.75,
            "ratio_gout": 0.75,
        }
        downsample_conv_kwargs = {
            "ratio_gin": 0.0,
            "ratio_gout": 0.0,
        }

        model = [
            nn.ReflectionPad2d(3),
            INIT_FFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0)
        ]

        ### downsample
        for i in range(n_downsampling):  # n_downsampling -- 3
            mult = 2 ** i # ==> 1, 2, 4
            # print(f"ngf * mult = {ngf * mult}, ngf * mult * 2={ngf * mult * 2}")
            # ngf * mult = 64, ngf * mult * 2=128
            # ngf * mult = 128, ngf * mult * 2=256
            # ngf * mult = 256, ngf * mult * 2=512

            cur_conv_kwargs = dict(downsample_conv_kwargs)
            if i == n_downsampling - 1:
                cur_conv_kwargs["ratio_gout"] = resnet_conv_kwargs.get("ratio_gin", 0)

            # cur_conv_kwargs = {'ratio_gin': 0.0, 'ratio_gout': 0.0}
            # cur_conv_kwargs = {'ratio_gin': 0.0, 'ratio_gout': 0.0}
            # cur_conv_kwargs = {'ratio_gin': 0.0, 'ratio_gout': 0.75}

            if i == n_downsampling - 1:
                model += [
                    START_REST_FFC_BN_ACT(ngf * mult, ngf * mult * 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        **cur_conv_kwargs
                    )
                ]
            else:
                model += [
                    DOWN_FFC_BN_ACT(ngf * mult, ngf * mult * 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        **cur_conv_kwargs
                    )
                ]

        mult = 2 ** n_downsampling
        feats_num_bottleneck = ngf * mult
        ### resnet blocks
        for i in range(n_blocks):  # n_blocks -- 18
            cur_resblock = FFCResnetBlock(feats_num_bottleneck)
            model += [cur_resblock]

        model += [ReduceTupleLayer()]

        ### upsample
        for i in range(n_downsampling):  # n_downsampling -- 3
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.BatchNorm2d(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model.append(nn.Sigmoid())

        self.model = nn.Sequential(*model)

        self.load_weights()
        self.eval()

        # pdb.set_trace()

    def load_weights(self, model_path="models/image_patch.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path

        if not os.path.exists(checkpoint):
            raise IOError(f"Model checkpoint '{checkpoint}' doesn't exist.")

        # state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        state_dict = torch.load(checkpoint, map_location=torch.device("cpu"))
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        target_state_dict = self.state_dict()
        for n, p in state_dict.items():
            if 'val_evaluator.scores' in n:
                continue
            if 'test_evaluator.scores' in n:
                continue

            n = n.replace("generator.", "")
            if n in target_state_dict.keys():
                target_state_dict[n].copy_(p)
            else:
                raise KeyError(n)


    def forward(self, input):
        # input.size() -- [1, 4, 1000, 1504], input[:, 3:4, :, :].mean() -- 0.1590
        B, C, H, W = input.size()
        assert C == 4  # Make input is Bx4xHxW
        pad_h = (self.MAX_TIMES - (H % self.MAX_TIMES)) % self.MAX_TIMES
        pad_w = (self.MAX_TIMES - (W % self.MAX_TIMES)) % self.MAX_TIMES

        input = F.pad(input, (0, pad_w, 0, pad_h), 'reflect')

        # input
        input_mask = input[:, 3:4, :, :]
        input_bin_mask = (input_mask < 0.9).float().to(input.device)
        input_content = input[:, 0:3, :, :] * (1.0 - input_bin_mask)
        input_tensor = torch.cat((input_content, input_bin_mask), dim=1)

        # process
        # todos.debug.output_var("input_tensor", input_tensor)
        output_tensor = self.model(input_tensor)  # [1, 3, 1000, 1504]
        # todos.debug.output_var("input_tensor", output_tensor)

        # output
        output_tensor = output_tensor[:, :, 0:H, 0:W].clamp(0.0, 1.0)
        output_mask = torch.ones(B, 1, H, W).to(input.device)

        output = torch.cat((output_tensor, output_mask), dim=1)
        return output[:, :, 0:H, 0:W]
