# Fast Fourier Convolution NeurIPS 2020
# original implementation https://github.com/pkumivision/FFC/blob/main/model_zoo/ffc.py
# paper https://proceedings.neurips.cc/paper/2020/file/2fd5d41ec6cfab47e32164d5624269b1-Paper.pdf

import os
import torch
import torch.nn as nn
import math
from torch.autograd import Function

from typing import Tuple

import pdb

# https://zhuanlan.zhihu.com/p/653745531
# https://github.com/onnx/onnx/issues/4845
class OnnxRfft(Function):
    """Auto-grad function to mimic rfft for ONNX exporting
    """
    @staticmethod
    def forward(ctx, input, dim: int):
        y = torch.fft.rfft(input, dim=dim, norm="backward")
        return torch.view_as_real(y)

    @staticmethod
    def symbolic(g: torch.Graph, input: torch.Value, dim: int) -> torch.Value:
        # See https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.Rfft
        output = g.op(
            "com.microsoft::Rfft",
            input,
            normalized_i=0,
            onesided_i=1,
            signal_ndim_i=dim,
        )

        return output
onnx_rfft = OnnxRfft.apply

class OnnxFft(Function):
    """Auto-grad function to mimic rfft for ONNX exporting
    """
    @staticmethod
    def forward(ctx, input, dim: int):
        # We need to mimic the behavior of Contrib RFFT which assumes
        # DFT of last dim and no normalization.
        y = torch.fft.fft(input, dim=dim, norm="backward")
        return y # torch.view_as_real(y)

    @staticmethod
    def symbolic(g: torch.Graph, input: torch.Value, dim: int) -> torch.Value:
        # See https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.Rfft
        output = g.op(
            "com.microsoft::Rfft",
            input,
            normalized_i=0,
            onesided_i=0, # same as OnnxRfft, but two-side ouput
            signal_ndim_i=dim,
        )

        return output
onnx_fft=OnnxFft.apply

class OnnxIrfft(Function):
    """Auto-grad function to mimic irfft for ONNX exporting
    """
    @staticmethod
    def forward(ctx, input, dim: int):
        # input is complex64
        return torch.fft.irfft(input, dim=dim, norm="backward") # output -- real

    @staticmethod
    def symbolic(g: torch.Graph, input: torch.Value) -> torch.Value:
        """Symbolic representation for onnx graph"""
        return g.op(
                "com.microsoft::Irfft",
                input,
                normalized_i=0,
                onesided_i=1, 
                signal_ndim_i=dim,
            )
onnx_irfft=OnnxIrfft.apply

class OnnxIfft(Function):
    """Auto-grad function to mimic irfft for ONNX exporting
    """
    @staticmethod
    def forward(ctx, input, dim:int):
        # return torch.fft.ifft(torch.view_as_complex(input), dim=dim, norm="backward")
        return torch.fft.ifft(input, dim=dim, norm="backward")

    @staticmethod
    def symbolic(g: torch.Graph, input: torch.Value, dim:int) -> torch.Value:
        """Symbolic representation for onnx graph"""
        return g.op(
                "com.microsoft::Irfft",
                input,
                normalized_i=0,
                onesided_i=0, # same as OnnxIrfft, two-side ouput
                signal_ndim_i=dim,
            )
onnx_ifft=OnnxIfft.apply

class OnnxComplex(Function):
    """Auto-grad function to mimic irfft for ONNX exporting
    """
    @staticmethod
    def forward(ctx, input1, input2):
        return torch.complex(input1, input2)

    @staticmethod
    def symbolic(g: torch.Graph, input1: torch.Value, input2: torch.Value) -> torch.Value:
        """Symbolic representation for onnx graph"""
        return g.op("Concat", input1, input2, axis_i=-1)
onnx_complex = OnnxComplex.apply


def rfftn(input, dim: Tuple[int, int]):
    D2, D3 = dim
    # input is real, size() -- [1, 192, 125, 188]
    S2:int = input.shape[D2]
    S3:int = input.shape[D3]

    output = onnx_fft(input, D2) # as rfft but two-side ouput, complex64
    output = onnx_fft(output, D3) # complex64
    output = torch.index_select(output, D3, torch.arange(S3//2+1).to(input.device))

    return output/math.sqrt(S2)/math.sqrt(S3) # output.dtype is torch.complex64, size() -- 1, 192, 125, 95]


def irfftn(input, sl: Tuple[int, int], dim: Tuple[int, int]):
    H, W = sl
    D2, D3 = dim
    # input is torch.complex64, size() -- [1, 192, 125, 95]
    output = onnx_ifft(input, D2) # torch.complex64, torch.Size([1, 192, 125, 95]
    output = onnx_irfft(output, D3) # torch.complex64, torch.Size([1, 192, 125, 95]

    return output*math.sqrt(H)*math.sqrt(W) # torch.float32, torch.Size([1, 192, 125, 188]

class FourierUnit(nn.Module):
    '''2D Fourier'''
    def __init__(self, in_channels, out_channels, groups=1):
        super().__init__()
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
        self.relu = nn.ReLU(inplace=True)
        # FourierUnit(
        #   (conv_layer): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #   (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #   (relu): ReLU(inplace=True)
        # )

    def forward(self, x):
        B, C, H, W = x.size()

        # x.size() -- [1, 192, 125, 188]
        # ffted = torch.fft.rfftn(x, dim=(2, 3), norm="ortho")
        ffted = rfftn(x, dim=(2, 3))
        # ffted.size() -- [1, 192, 125, 95], torch.complex64

        # ffted = torch.stack((ffted.real, ffted.imag), dim=-1) # [1, 192, 125, 95, 2]
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1) # [1, 192, 125, 95, 2]

        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # [1, 192, 2, 125, 95], (B, c, 2, h, w/2+1)
        ffted = ffted.view((B, -1, ) + ffted.size()[3:]) # [1, 384, 125, 95]

        ffted = self.conv_layer(ffted)  # [1, 384, 125, 95] (B, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = (
            ffted.view((B, -1, 2, ) + ffted.size()[2:])
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        # [1, 192, 2, 125, 95] ==> [1, 192, 125, 95, 2], (B, c, t, h, w/2+1, 2)
        # ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        ffted = onnx_complex(ffted[..., 0], ffted[..., 1])

        # output = torch.fft.irfftn(ffted, s=(H, W), dim=(2, 3), norm="ortho")
        # output = torch.fft.irfftn(ffted, dim=(2, 3), norm="ortho")
        output = irfftn(ffted, sl=(H, W), dim=(2, 3))

        return output # [1, 192, 125, 188]


class SpectralTransform(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super().__init__()

        if stride == 2:
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

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.convl2l = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, 
            dilation, groups, bias, padding_mode="reflect"
        )

    def forward(self, x) -> Tuple[torch.Tensor, int]:
        out_xl = self.convl2l(x)
        return (out_xl, 0)


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

        self.ffc = INIT_FFC(
            in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, stride, padding, 
            dilation, groups, bias
        )
        self.bn_l = nn.BatchNorm2d(out_channels)
        self.act_l = nn.ReLU()

    def forward(self, x) -> Tuple[torch.Tensor, int]:
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))

        return (x_l, x_g)


class DOWN_FFC(nn.Module):
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
        assert stride == 1 or stride == 2, "Stride should be 1 or 2."

        out_channels = out_channels - int(out_channels * ratio_gout)
        self.convl2l = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, 
            dilation, groups, bias, padding_mode="reflect"
        )

    def forward(self, x: Tuple[torch.Tensor, int]) -> Tuple[torch.Tensor, int]:
        out_xl = self.convl2l(x[0])
        return (out_xl, 0)


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

        self.ffc = DOWN_FFC(
            in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, 
            stride, padding, dilation, groups, bias
        )
        out_channels = out_channels - int(out_channels * ratio_gout)
        self.bn_l = nn.BatchNorm2d(out_channels)
        self.act_l = nn.ReLU()

    def forward(self, x: Tuple[torch.Tensor, int]) -> Tuple[torch.Tensor, int]:
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))

        return (x_l, x_g)


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

        self.ffc = START_REST_FFC(
            in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, 
            stride, padding, dilation, groups, bias
        )
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = nn.BatchNorm2d(out_channels - global_channels)
        self.bn_g = nn.BatchNorm2d(global_channels)
        self.act_l = nn.ReLU()
        self.act_g = nn.ReLU()

    def forward(self, x: Tuple[torch.Tensor, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))

        return (x_l, x_g)


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
        assert stride == 1 or stride == 2, "Stride should be 1 or 2."

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

        self.ffc = REST_FFC(
            in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, 
            stride, padding, dilation, groups, bias
        )
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = nn.BatchNorm2d(out_channels - global_channels)
        self.bn_g = nn.BatchNorm2d(global_channels)
        self.act_l = nn.ReLU()
        self.act_g = nn.ReLU()

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))

        return (x_l, x_g)


class FFCResnetBlock(nn.Module):
    """FFC -- Fast Fourier Convolution"""

    def __init__(self, dim, dilation=1):
        super().__init__()
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


class ReduceTupleLayer(nn.Module):
    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        return torch.cat(x, dim=1)


class FFCResNetGenerator(nn.Module):
    """FFC -- Fast Fourier Convolution"""
    def __init__(self,
        input_nc=4,
        output_nc=3,
        ngf=64,
        n_downsampling=3,
        n_blocks=18,
        max_features=1024,
    ):
        super().__init__()
        self.MAX_H = 1024
        self.MAX_W = 1024
        self.MAX_TIMES = 4
        # Define max GPU/CPU memory -- 5G(1024x2048), 530ms

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
            mult = 2 ** i

            cur_conv_kwargs = dict(downsample_conv_kwargs)
            if i == n_downsampling - 1:
                cur_conv_kwargs["ratio_gout"] = resnet_conv_kwargs.get("ratio_gin", 0)

            if i == n_downsampling - 1:
                model += [
                    START_REST_FFC_BN_ACT(
                        min(max_features, ngf * mult),
                        min(max_features, ngf * mult * 2),
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        **cur_conv_kwargs
                    )
                ]
            else:
                model += [
                    DOWN_FFC_BN_ACT(
                        min(max_features, ngf * mult),
                        min(max_features, ngf * mult * 2),
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        **cur_conv_kwargs
                    )
                ]

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)
        ### resnet blocks
        for i in range(n_blocks):  # n_blocks -- 18
            cur_resblock = FFCResnetBlock(feats_num_bottleneck)

            model += [cur_resblock]

        model += [ReduceTupleLayer()]

        ### upsample
        for i in range(n_downsampling):  # n_downsampling -- 3
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    min(max_features, ngf * mult),
                    min(max_features, int(ngf * mult / 2)),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.BatchNorm2d(min(max_features, int(ngf * mult / 2))),
                nn.ReLU(True),
            ]

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model.append(nn.Sigmoid())

        self.model = nn.Sequential(*model)

        self.load_weights()
        self.eval()


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

        # input
        input_mask = input[:, 3:4, :, :]
        input_bin_mask = (input_mask < 0.9).float().to(input.device)
        input_content = input[:, 0:3, :, :] * (1.0 - input_bin_mask)
        input_tensor = torch.cat((input_content, input_bin_mask), dim=1)

        # process
        output_tensor = self.model(input_tensor)  # [1, 3, 1000, 1504]

        # output
        output_tensor = output_tensor[:, :, 0:H, 0:W].clamp(0.0, 1.0)
        output_mask = torch.ones(B, 1, H, W).to(input.device)

        return torch.cat((output_tensor, output_mask), dim=1)
