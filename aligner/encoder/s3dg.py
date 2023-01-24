# Initially copied from the MIL-NCE repo.
"""Contains the definition for Gated Separable 3D network (S3D-G). """
from typing import Literal, Tuple

import torch
from overrides import overrides
from torch import nn
from torch.nn.common_types import _size_3_t, _size_6_t


class InceptionBlock(nn.Module):
    def __init__(self, input_dim: int, num_outputs_0_0a: int, num_outputs_1_0a: int, num_outputs_1_0b: int,
                 num_outputs_2_0a: int, num_outputs_2_0b: int, num_outputs_3_0b: int, gating: bool = True) -> None:
        super().__init__()
        self.conv_b0 = STConv3D(input_dim, num_outputs_0_0a, kernel_size=1)
        self.conv_b1_a = STConv3D(input_dim, num_outputs_1_0a, kernel_size=1)
        self.conv_b1_b = STConv3D(num_outputs_1_0a, num_outputs_1_0b, kernel_size=3, padding=1, separable=True)
        self.conv_b2_a = STConv3D(input_dim, num_outputs_2_0a, kernel_size=1)
        self.conv_b2_b = STConv3D(num_outputs_2_0a, num_outputs_2_0b, kernel_size=3, padding=1, separable=True)
        self.maxpool_b3 = torch.nn.MaxPool3d(kernel_size=3, stride=1, padding=1)
        self.conv_b3_b = STConv3D(input_dim, num_outputs_3_0b, 1)
        self.gating = gating
        self.output_dim = num_outputs_0_0a + num_outputs_1_0b + num_outputs_2_0b + num_outputs_3_0b
        if gating:
            self.gating_b0 = SelfGating(num_outputs_0_0a)
            self.gating_b1 = SelfGating(num_outputs_1_0b)
            self.gating_b2 = SelfGating(num_outputs_2_0b)
            self.gating_b3 = SelfGating(num_outputs_3_0b)

    @overrides(check_signature=False)
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        b0 = self.conv_b0(input_)
        b1 = self.conv_b1_a(input_)
        b1 = self.conv_b1_b(b1)
        b2 = self.conv_b2_a(input_)
        b2 = self.conv_b2_b(b2)
        b3 = self.maxpool_b3(input_)
        b3 = self.conv_b3_b(b3)
        if self.gating:
            b0 = self.gating_b0(b0)
            b1 = self.gating_b1(b1)
            b2 = self.gating_b2(b2)
            b3 = self.gating_b3(b3)
        return torch.cat((b0, b1, b2, b3), dim=1)


class SelfGating(nn.Module):
    """Feature gating as used in S3D-G. """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, input_dim)
        self.sigmoid = nn.modules.activation.Sigmoid()

    @overrides(check_signature=False)
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        spatiotemporal_average = input_.mean(dim=[2, 3, 4])
        weights = self.fc(spatiotemporal_average)
        weights = self.sigmoid(weights)
        return weights[:, :, None, None, None] * input_


def _size3_to_spatial_temporal(size: _size_3_t, fill_value: int) -> Tuple[_size_3_t, _size_3_t]:
    size = nn.modules.conv._triple(size)
    return (fill_value, size[1], size[2]), (size[0], fill_value, fill_value)


class STConv3D(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, kernel_size: _size_3_t, stride: _size_3_t = 1,
                 padding: _size_3_t = 0, separable: bool = False) -> None:
        super().__init__()
        self.separable = separable
        self.relu = nn.ReLU(inplace=True)

        if separable:
            assert (isinstance(kernel_size, int) and kernel_size != 1) or kernel_size[0] != 1

            spatial_kernel_size, temporal_kernel_size = _size3_to_spatial_temporal(kernel_size, fill_value=1)
            spatial_stride, temporal_stride = _size3_to_spatial_temporal(stride, fill_value=1)
            spatial_padding, temporal_padding = _size3_to_spatial_temporal(padding, fill_value=0)

            self.conv1 = nn.Conv3d(input_dim, output_dim, kernel_size=spatial_kernel_size, stride=spatial_stride,
                                   padding=spatial_padding, bias=False)
            self.conv2 = nn.Conv3d(output_dim, output_dim, kernel_size=temporal_kernel_size, stride=temporal_stride,
                                   padding=temporal_padding, bias=False)
            self.bn2 = nn.BatchNorm3d(output_dim)
        else:
            self.conv1 = nn.Conv3d(input_dim, output_dim, kernel_size=kernel_size, stride=stride,  # noqa
                                   padding=padding, bias=False)

        self.bn1 = nn.BatchNorm3d(output_dim)

    @overrides(check_signature=False)
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(input_)))
        if self.separable:
            out = self.relu(self.bn2(self.conv2(out)))
        return out


def _pad_top_bottom(kernel_dim: int, stride_val: int) -> Tuple[int, int]:
    pad_along = max(kernel_dim - stride_val, 0)
    pad_top_ = pad_along // 2
    pad_bottom_ = pad_along - pad_top_
    return pad_top_, pad_bottom_


def _get_padding_shape(kernel_size: _size_3_t, stride: _size_3_t) -> _size_6_t:
    kernel_size = nn.modules.conv._triple(kernel_size)
    stride = nn.modules.conv._triple(stride)

    padding_shape = [padding_value
                     for pair in zip(kernel_size, stride)
                     for padding_value in _pad_top_bottom(*pair)]

    depth_top = padding_shape.pop(0)
    depth_bottom = padding_shape.pop(0)
    padding_shape.append(depth_top)
    padding_shape.append(depth_bottom)

    return tuple(padding_shape)


class MaxPool3dTFPadding(torch.nn.Module):
    def __init__(self, kernel_size: _size_3_t, stride: _size_3_t, padding: Literal["SAME"] = "SAME") -> None:
        super().__init__()

        if padding == "SAME":
            self.padding_shape = _get_padding_shape(kernel_size, stride)
            self.pad = torch.nn.ConstantPad3d(self.padding_shape, 0)
        else:
            raise ValueError(f"Padding strategy not supported: {padding}")

        self.pool = torch.nn.MaxPool3d(kernel_size, stride, ceil_mode=True)

    @overrides(check_signature=False)
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        input_ = self.pad(input_)
        return self.pool(input_)


class S3DG(nn.Module):
    def __init__(self, embedding_size: int = 512, space_to_depth: bool = True,
                 init: Literal["default", "kaiming_normal"] = "default", use_last_layer: bool = True) -> None:
        super().__init__()
        self.use_last_layer = use_last_layer
        self.space_to_depth = space_to_depth
        if space_to_depth:
            self.conv1 = STConv3D(24, 64, kernel_size=(2, 4, 4), stride=1, padding=(1, 2, 2), separable=False)  # noqa
        else:
            self.conv1 = STConv3D(3, 64, kernel_size=(3, 7, 7), stride=2, padding=(1, 3, 3), separable=False)  # noqa
        self.conv_2b = STConv3D(64, 64, kernel_size=1, separable=False)
        self.conv_2c = STConv3D(64, 192, kernel_size=3, padding=1, separable=True)
        self.gating = SelfGating(192)
        self.maxpool_2a = MaxPool3dTFPadding(kernel_size=(1, 3, 3), stride=(1, 2, 2))
        self.maxpool_3a = MaxPool3dTFPadding(kernel_size=(1, 3, 3), stride=(1, 2, 2))
        self.mixed_3b = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.mixed_3c = InceptionBlock(self.mixed_3b.output_dim, 128, 128, 192, 32, 96, 64)
        self.maxpool_4a = MaxPool3dTFPadding(kernel_size=3, stride=2)
        self.mixed_4b = InceptionBlock(self.mixed_3c.output_dim, 192, 96, 208, 16, 48, 64)
        self.mixed_4c = InceptionBlock(self.mixed_4b.output_dim, 160, 112, 224, 24, 64, 64)
        self.mixed_4d = InceptionBlock(self.mixed_4c.output_dim, 128, 128, 256, 24, 64, 64)
        self.mixed_4e = InceptionBlock(self.mixed_4d.output_dim, 112, 144, 288, 32, 64, 64)
        self.mixed_4f = InceptionBlock(self.mixed_4e.output_dim, 256, 160, 320, 32, 128, 128)
        self.maxpool_5a = self.maxPool3d_5a_2x2 = MaxPool3dTFPadding(kernel_size=2, stride=2)
        self.mixed_5b = InceptionBlock(self.mixed_4f.output_dim, 256, 160, 320, 32, 128, 128)
        self.mixed_5c = InceptionBlock(self.mixed_5b.output_dim, 384, 192, 384, 48, 128, 128)
        self.fc = nn.Linear(self.mixed_5c.output_dim, embedding_size)

        if init == "kaiming_normal":
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                elif isinstance(m, nn.BatchNorm3d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    @property
    def output_size(self) -> int:
        return self.fc.out_features if self.use_last_layer else self.mixed_5c.output_dim

    @staticmethod
    def _space_to_depth(input_: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = input_.shape
        input_ = input_.view(B, C, T // 2, 2, H // 2, 2, W // 2, 2)
        input_ = input_.permute(0, 3, 5, 7, 1, 2, 4, 6)
        input_ = input_.contiguous().view(B, 8 * C, T // 2, H // 2, W // 2)
        return input_

    @overrides(check_signature=False)
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if self.space_to_depth:
            input_ = self._space_to_depth(input_)
        net = self.conv1(input_)
        if self.space_to_depth:
            net = net[:, :, 1:, 1:, 1:]
        net = self.maxpool_2a(net)
        net = self.conv_2b(net)
        net = self.conv_2c(net)
        if self.gating:
            net = self.gating(net)
        net = self.maxpool_3a(net)
        net = self.mixed_3b(net)
        net = self.mixed_3c(net)
        net = self.maxpool_4a(net)
        net = self.mixed_4b(net)
        net = self.mixed_4c(net)
        net = self.mixed_4d(net)
        net = self.mixed_4e(net)
        net = self.mixed_4f(net)
        net = self.maxpool_5a(net)
        net = self.mixed_5b(net)
        net = self.mixed_5c(net)
        net = torch.mean(net, dim=(2, 3, 4))
        if self.use_last_layer:
            return self.fc(net)
        else:
            return net
