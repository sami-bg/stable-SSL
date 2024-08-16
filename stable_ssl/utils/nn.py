from typing import Tuple

import torch
import torchvision
import torch.nn as nn


def load_model_without_classifier(name: str, **kwargs) -> Tuple[nn.Module, int]:
    if name == "resnet9":
        model = resnet9(**kwargs)
    elif name == "ConvMixer":
        model = ConvMixer(**kwargs)
    else:
        model = torchvision.models.__dict__[name](**kwargs)

    if "alexnet" in name:
        fan_in = model.classifier[6].in_features
        model.classifier[6] = nn.Identity()
    elif "convnext" in name:
        fan_in = model.classifier[2].in_features
        model.classifier[2] = nn.Identity()
    elif "convnext" in name:
        fan_in = model.classifier[2].in_features
        model.classifier[2] = nn.Identity()
    elif (
        "resnet" in name or "resnext" in name or "regnet" in name or name == "ConvMixer"
    ):
        fan_in = model.fc.in_features
        model.fc = nn.Identity()
    elif "densenet" in name:
        fan_in = model.classifier.in_features
        model.classifier = nn.Identity()
    elif "mobile" in name:
        fan_in = model.classifier[-1].in_features
        model.classifier[-1] = nn.Identity()
    elif "vit" in name:
        fan_in = model.heads.head.in_features
        model.heads.head = nn.Identity()
    elif "swin" in name:
        fan_in = model.head.in_features
        model.head = nn.Identity()
    return model, fan_in


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
