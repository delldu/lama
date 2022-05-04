# Fast Fourier Convolution NeurIPS 2020
# original implementation https://github.com/pkumivision/FFC/blob/main/model_zoo/ffc.py
# paper https://proceedings.neurips.cc/paper/2020/file/2fd5d41ec6cfab47e32164d5624269b1-Paper.pdf

import torch
import torch.nn as nn
from typing import Tuple

import pdb


class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, ffc3d=False):
        super(FourierUnit, self).__init__()
        self.groups = groups

        self.conv_layer = torch.nn.Conv2d(
            in_channels=in_channels * 2,
            out_channels=out_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=self.groups,
            bias=False,
        )
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

        self.ffc3d = ffc3d

    def forward(self, x):
        batch = x.shape[0]

        # fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm="ortho")
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view(
            (
                batch,
                -1,
            )
            + ffted.size()[3:]
        )

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2, ) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()
        # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        # ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        ifft_shape_slice = x.shape[-2:]

        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm="ortho")

        return output


class SpectralTransform(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):

        super(SpectralTransform, self).__init__()

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

        self.conv2 = torch.nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)
        output = self.conv2(x + output)

        return output


class INIT_FFC(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        ratio_gin=0.0,
        ratio_gout=0.0,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super(INIT_FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.convl2l = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode="reflect"
        )

    def forward(self, x) -> Tuple[torch.Tensor, int]:
        out_xl = self.convl2l(x)
        return (out_xl, 0)


class INIT_FFC_BN_ACT(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        ratio_gin=0.0,
        ratio_gout=0.0,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        norm_layer=nn.BatchNorm2d,
        activation_layer=nn.Identity,
    ):
        super(INIT_FFC_BN_ACT, self).__init__()

        self.ffc = INIT_FFC(
            in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, stride, padding, dilation, groups, bias
        )
        self.bn_l = norm_layer(out_channels)
        self.act_l = activation_layer(inplace=True)

    def forward(self, x) -> Tuple[torch.Tensor, int]:
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))

        return (x_l, x_g)


class DOWN_FFC(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        ratio_gin=0.0,
        ratio_gout=0.75,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super(DOWN_FFC, self).__init__()
        assert stride == 1 or stride == 2, "Stride should be 1 or 2."

        out_channels = out_channels - int(out_channels * ratio_gout)
        self.convl2l = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode="reflect"
        )

    def forward(self, x: Tuple[torch.Tensor, int]) -> Tuple[torch.Tensor, int]:
        out_xl = self.convl2l(x[0])
        return (out_xl, 0)


class DOWN_FFC_BN_ACT(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        ratio_gin=0.0,
        ratio_gout=0.75,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        norm_layer=nn.BatchNorm2d,
        activation_layer=nn.Identity,
    ):
        super(DOWN_FFC_BN_ACT, self).__init__()

        self.ffc = DOWN_FFC(
            in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, stride, padding, dilation, groups, bias
        )
        out_channels = out_channels - int(out_channels * ratio_gout)
        self.bn_l = norm_layer(out_channels)
        self.act_l = activation_layer(inplace=True)

    def forward(self, x: Tuple[torch.Tensor, int]) -> Tuple[torch.Tensor, int]:
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))

        return (x_l, x_g)


class START_REST_FFC(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        ratio_gin=0.75,
        ratio_gout=0.75,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super(START_REST_FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        self.convl2l = nn.Conv2d(
            in_cl, out_cl, kernel_size, stride, padding, dilation, groups, bias, padding_mode="reflect"
        )
        self.convl2g = nn.Conv2d(
            in_cl, out_cg, kernel_size, stride, padding, dilation, groups, bias, padding_mode="reflect"
        )

    def forward(self, x: Tuple[torch.Tensor, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        x_l, x_g = x
        out_xl = self.convl2l(x_l)
        out_xg = self.convl2g(x_l)
        return (out_xl, out_xg)


class START_REST_FFC_BN_ACT(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        ratio_gin=0.75,
        ratio_gout=0.75,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        norm_layer=nn.BatchNorm2d,
        activation_layer=nn.Identity,
    ):
        super(START_REST_FFC_BN_ACT, self).__init__()

        self.ffc = START_REST_FFC(
            in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, stride, padding, dilation, groups, bias
        )
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = norm_layer(out_channels - global_channels)
        self.bn_g = norm_layer(global_channels)
        self.act_l = activation_layer(inplace=True)
        self.act_g = activation_layer(inplace=True)

    def forward(self, x: Tuple[torch.Tensor, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))

        return (x_l, x_g)


class REST_FFC(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        ratio_gin=0.75,
        ratio_gout=0.75,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super(REST_FFC, self).__init__()
        assert stride == 1 or stride == 2, "Stride should be 1 or 2."

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        self.convl2l = nn.Conv2d(
            in_cl, out_cl, kernel_size, stride, padding, dilation, groups, bias, padding_mode="reflect"
        )
        self.convl2g = nn.Conv2d(
            in_cl, out_cg, kernel_size, stride, padding, dilation, groups, bias, padding_mode="reflect"
        )
        self.convg2l = nn.Conv2d(
            in_cg, out_cl, kernel_size, stride, padding, dilation, groups, bias, padding_mode="reflect"
        )
        self.convg2g = SpectralTransform(in_cg, out_cg, stride, 1 if groups == 1 else groups // 2)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x_l, x_g = x
        out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        out_xg = self.convl2g(x_l) + self.convg2g(x_g)
        return (out_xl, out_xg)


class REST_FFC_BN_ACT(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        ratio_gin=0.75,
        ratio_gout=0.75,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        norm_layer=nn.BatchNorm2d,
        activation_layer=nn.Identity,
    ):
        super(REST_FFC_BN_ACT, self).__init__()

        self.ffc = REST_FFC(
            in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, stride, padding, dilation, groups, bias
        )
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = norm_layer(out_channels - global_channels)
        self.bn_g = norm_layer(global_channels)

        self.act_l = activation_layer(inplace=True)
        self.act_g = activation_layer(inplace=True)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))

        return (x_l, x_g)


class FFCResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, activation_layer=nn.ReLU, dilation=1):
        super().__init__()
        self.conv1 = REST_FFC_BN_ACT(
            dim, dim,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )
        self.conv2 = REST_FFC_BN_ACT(
            dim, dim,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
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
    def __init__(
        self,
        input_nc=4,
        output_nc=3,
        ngf=64,
        n_downsampling=3,
        n_blocks=18,
        norm_layer=nn.BatchNorm2d,
        activation_layer=nn.ReLU,
        max_features=1024,
    ):
        assert n_blocks >= 0
        super().__init__()

        init_conv_kwargs = {
            "ratio_gin": 0.0,
            "ratio_gout": 0.0,
        }
        resnet_conv_kwargs = {
            "ratio_gin": 0.75,
            "ratio_gout": 0.75,
        }
        downsample_conv_kwargs = {
            "ratio_gin": init_conv_kwargs["ratio_gout"],
            "ratio_gout": init_conv_kwargs["ratio_gin"],  # resnet_conv_kwargs['ratio_gin']
        }

        # INIT_FFC_BN_ACT (0.0, 0.0)
        # DOWN_FFC_BN_ACT (0.0, 0.75)
        # REST_FFC_BN_ACT (0.75, 0.75)
        model = [
            nn.ReflectionPad2d(3),
            INIT_FFC_BN_ACT(
                input_nc, ngf, kernel_size=7, padding=0, norm_layer=norm_layer, activation_layer=activation_layer
            ),
        ]

        ### downsample
        for i in range(n_downsampling):  # n_downsampling -- 3
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs["ratio_gout"] = resnet_conv_kwargs.get("ratio_gin", 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs

            if i == n_downsampling - 1:
                model += [
                    START_REST_FFC_BN_ACT(
                        min(max_features, ngf * mult),
                        min(max_features, ngf * mult * 2),
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        norm_layer=norm_layer,
                        activation_layer=activation_layer,
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
                        norm_layer=norm_layer,
                        activation_layer=activation_layer,
                        **cur_conv_kwargs
                    )
                ]

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)
        ### resnet blocks
        for i in range(n_blocks):  # n_blocks -- 18
            cur_resblock = FFCResnetBlock(
                feats_num_bottleneck, activation_layer=activation_layer, norm_layer=norm_layer
            )
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

    def forward(self, input):
        # input.size() -- [1, 4, 1000, 1504], input[:, 3:4, :, :].mean() -- 0.1590
        output = self.model(input)
        # output.size() -- [1, 3, 1000, 1504]
        return output


if __name__ == "__main__":
    model = FFCResNetGenerator()
    model = torch.jit.script(model)

    pdb.set_trace()
