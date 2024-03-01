# coding=utf-8
import torch.nn as nn
from torchvision import models
import timm
import torch
import numpy as np

eff_dict = {
    "effb0": timm.create_model('efficientnet_b0', pretrained=True),
}

class EffBase(nn.Module):
    def __init__(self, eff_name):
        super(EffBase, self).__init__()
        model_eff = eff_dict[eff_name]
        self.conv_stem = model_eff.conv_stem
        self.bn1 = model_eff.bn1
        self.act1 = model_eff.act1
        self.blocks = model_eff.blocks
        self.conv_head = model_eff.conv_head
        self.bn2 = model_eff.bn2
        self.act2 = model_eff.act2
        self.global_pool = model_eff.global_pool
        self.in_features = model_eff.classifier.in_features

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.global_pool(x)
        return x