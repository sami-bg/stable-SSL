"""
File: nn.py
Project: torchstrap
-----
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""

from torch import nn
import torch
from torch.jit import ScriptModule
import numpy as np
from typing import Tuple


class CategoricalConditionalBatchNorm2d(ScriptModule):
    def __init__(self, num_features: int, num_conditions: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_conditions, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x: torch.Tensor, condition: torch.Tensor):
        out = self.bn(x)
        gamma, beta = self.embed(condition).chunk(2, 1)
        out = gamma.view(-1, 1, 1) * out + beta.view(-1, 1, 1)
        return out


class ConditionalBatchNorm2d(ScriptModule):
    def __init__(self, num_features: int, num_conditioning_features: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Linear(num_conditioning_features, num_features * 2, bias=False)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x: torch.Tensor, condition: torch.Tensor):
        out = self.bn(x)
        gamma, beta = self.embed(condition).chunk(2, 1)
        out = gamma.view(-1, 1, 1) * out + beta.view(-1, 1, 1)
        return out


class ConvMixer(nn.Module):
    # https://openreview.net/forum?id=TVHS5Y4dNvM
    def __init__(
        self,
        in_channels=3,
        num_classes=10,
        dim=64,
        depth=6,
        kernel_size=9,
        patch_size=7,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )

        self.blocks_a = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.BatchNorm2d(dim),
                    nn.ReLU(),
                )
                for _ in range(depth)
            ]
        )
        self.blocks_b = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=1), nn.BatchNorm2d(dim), nn.ReLU()
                )
                for _ in range(depth)
            ]
        )

        self.pool = nn.Sequential(nn.AdaptiveMaxPool2d(1), nn.Flatten())
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, xb):
        out = self.conv1(xb)
        for a, b in zip(self.blocks_a, self.blocks_b):
            out = out + a(out)
            out = b(out)
        out = self.fc(self.pool(out))
        return out


class resnet9(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()

        self.conv1 = self.conv_block(in_channels, 64)
        self.conv2 = self.conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(self.conv_block(128, 128), self.conv_block(128, 128))

        self.conv3 = self.conv_block(128, 256, pool=True)
        self.conv4 = self.conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(self.conv_block(512, 512), self.conv_block(512, 512))
        self.conv5 = self.conv_block(512, 1028, pool=True)
        self.res3 = nn.Sequential(
            self.conv_block(1028, 1028), self.conv_block(1028, 1028)
        )
        self.pool = nn.Sequential(nn.AdaptiveMaxPool2d(1), nn.Flatten())

        self.fc = nn.Linear(1028, num_classes)

    @staticmethod
    def conv_block(in_channels, out_channels, pool=False):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.conv5(out)
        out = self.res3(out) + out
        out = self.fc(self.pool(out))
        return out


class FullyConnected(nn.Module):

    activation = nn.ReLU

    def __init__(
        self,
        in_shape,
        out_shape,
        widths,
        batch_norm=True,
        residual=False,
        linear_last_layer=True,
    ):
        super().__init__()
        if not hasattr(in_shape, "__len__"):
            in_shape = (in_shape,)
        if not hasattr(out_shape, "__len__"):
            out_shape = (out_shape,)

        widths = [np.prod(in_shape)] + list(widths) + [np.prod(out_shape)]
        encoder = [nn.Flatten()]
        for fin, fout in zip(widths[:-1], widths[1:]):
            encoder.append(self.block(fin, fout, batch_norm, residual))
        if linear_last_layer:
            encoder[-1][1] = nn.Identity()
            encoder[-1][2] = nn.Identity()
        encoder.append(nn.Unflatten(1, out_shape))
        self.encoder = nn.Sequential(*encoder)

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded

    def block(self, in_features, out_features, batch_norm, residual):
        linear = nn.Linear(in_features, out_features, bias=not batch_norm)
        bn = nn.BatchNorm1d(out_features) if batch_norm else nn.Identity()
        return torch.nn.Sequential(linear, bn, self.activation())


class SmoothReLU(nn.Module):
    def __init__(self, smoothness=0, in_features=1, trainable=False):
        super().__init__()
        param = torch.nn.Parameter(torch.zeros(in_features) + smoothness)
        param.requires_grad_(trainable)
        self.register_parameter("smoothness", param)

    def forward(self, x):
        return torch.sigmoid(x / nn.functional.softplus(self.smoothness)) * x


class LazySmoothReLU(nn.modules.lazy.LazyModuleMixin, SmoothReLU):

    cls_to_become = SmoothReLU
    weight: nn.parameter.UninitializedParameter

    def __init__(self, axis=-1, device=None, dtype=None):
        super().__init__()
        if type(axis) not in [tuple, list]:
            axis = [axis]
        self.axis = axis
        self.smoothness = nn.parameter.UninitializedParameter(
            device=device, dtype=dtype
        )

    def initialize_parameters(self, input) -> None:
        if self.has_uninitialized_params():
            with torch.no_grad():
                s = [1 for _ in range(input.ndim)]
                for i in self.axis:
                    s[i] = input.size(i)
                self.smoothness.materialize(s)
                self.smoothness.copy_(torch.Tensor([0.541323]))


def load_without_classifier(name: str, **kwargs) -> Tuple[torch.nn.Module, int]:
    import torchvision

    if name == "resnet9":
        model = resnet9(**kwargs)
    elif name == "ConvMixer":
        model = ConvMixer(**kwargs)
    else:
        model = torchvision.models.__dict__[name](**kwargs)
    if "alexnet" in name:
        fan_in = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Identity()
    elif "convnext" in name:
        fan_in = model.classifier[2].in_features
        model.classifier[2] = torch.nn.Identity()
    elif "convnext" in name:
        fan_in = model.classifier[2].in_features
        model.classifier[2] = torch.nn.Identity()
    elif (
        "resnet" in name or "resnext" in name or "regnet" in name or name == "ConvMixer"
    ):
        fan_in = model.fc.in_features
        model.fc = torch.nn.Identity()
    elif "densenet" in name:
        fan_in = model.classifier.in_features
        model.classifier = torch.nn.Identity()
    elif "mobile" in name:
        fan_in = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Identity()
    elif "vit" in name:
        fan_in = model.heads.head.in_features
        model.heads.head = torch.nn.Identity()
    elif "swin" in name:
        fan_in = model.head.in_features
        model.head = torch.nn.Identity()
    return model, fan_in


def find_module(model: torch.nn.Module, module: torch.nn.Module):
    names = []
    values = []
    for child_name, child in model.named_modules():
        if isinstance(child, module):
            names.append(child_name)
            values.append(child)
    return names, values


def replace_module(model: torch.nn.Module, replacement_mapping):
    if not isinstance(model, torch.nn.Module):
        raise ValueError("Torch.nn.Module expected as input")
    for name, module in model.named_modules():
        if name == "":
            continue
        replacement = replacement_mapping(name, module)
        if replacement is None:
            continue
        module_names = name.split(".")
        # we go down the tree up to the parent
        parent = model
        for name in module_names[:-1]:
            parent = getattr(parent, name)
        setattr(parent, module_names[-1], replacement)
    return model
