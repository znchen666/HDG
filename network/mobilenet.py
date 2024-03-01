# coding=utf-8
import torch.nn as nn
from torchvision import models
import timm
import torch
import numpy as np

mob_dict = {
    "mobv2": models.mobilenet_v2(pretrained=True),
    "mobv3": timm.create_model('tf_mobilenetv3_small_075', pretrained=True),
    # "mobv3": timm.create_model('mobilenetv3_large_100', pretrained=True),
}

class MobBase(nn.Module):
    def __init__(self, mob_name):
        super(MobBase, self).__init__()
        model_mob = mob_dict[mob_name]
        self.conv_stem = model_mob.conv_stem
        self.bn1 = model_mob.bn1
        self.act1 = model_mob.act1
        self.blocks = model_mob.blocks
        self.global_pool = model_mob.global_pool
        self.conv_head = model_mob.conv_head
        self.act2 = model_mob.act2
        self.flatten = model_mob.flatten
        self.in_features = model_mob.classifier.in_features
        #
        # self.features = model_mob.features
        # self.in_features = 1280

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = self.flatten(x)
        return x

        # x = self.features(x)
        # return x