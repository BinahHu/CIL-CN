# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, avg_pool2d
from typing import List
import torchvision
import re


def conv3x3(in_planes: int, out_planes: int, stride: int=1) -> F.conv2d:
    """
    Instantiates a 3x3 convolutional layer with no bias.
    :param in_planes: number of input channels
    :param out_planes: number of output channels
    :param stride: stride of the convolution
    :return: convolutional layer
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """
    The basic block of ResNet.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int=1) -> None:
        """
        Instantiates the basic block of the network.
        :param in_planes: the number of input channels
        :param planes: the number of channels (to be possibly expanded)
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (10)
        """
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class Head(nn.Module):
    """
    CovNorm ResNet head
    """
    def __init__(self, nf: int, header_mode):
        super(Head, self).__init__()
        self.header_mode = header_mode

        if self.header_mode == 'small':
            self.conv1 = conv3x3(3, nf * 1)
        else:
            self.conv1 = nn.Conv2d(3, nf * 1, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(nf * 1)

    def forward(self, x):
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor list, (task_num, 10)
        """
        if self.header_mode == 'small':
            return relu(self.bn1(self.conv1(x)))
        else:
            x = F.interpolate(x, [256, 256], mode='bilinear')
            return self.maxpool(relu(self.bn1(self.conv1(x))))

class ResNet(nn.Module):
    """
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int, header_mode = 'small', args=None) -> None:
        """
        Instantiates the layers of the network.
        :param block: the basic ResNet block
        :param num_blocks: the number of blocks per layer
        :param num_classes: the number of output classes
        :param nf: the number of filters
        """
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.header = Head(nf, header_mode=header_mode)
        self.header_mode = header_mode
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)

        self._features = nn.Sequential(self.header,
                                       self.layer1,
                                       self.layer2,
                                       self.layer3,
                                       self.layer4
                                       )
        self.classifier = self.linear

        if args is not None and 'pretrained' in args['model']['selector'] and args['model']['selector']['pretrained'] is not None:
            pretrained_source = args['model']['selector']['pretrained']
            if pretrained_source == 'ImageNet':
                self.load_ImageNet_pretrained_model()
            elif pretrained_source == 'ImageNet-800':
                self.load_ImageNet800_pretrained_model()

    def load_ImageNet_pretrained_model(self):
        assert self.header_mode == 'big', "Only standard ResNet18 architecture can load ImageNet pretrained model"
        res18 = torchvision.models.resnet18(pretrained=True)
        res18_state = res18.state_dict()
        for name, p in self.named_parameters():
            if 'linear' not in name:
                para_name = name
                if 'header' in para_name:
                    para_name = re.match(r'header\.(.*)', para_name).group(1)
                if 'shortcut' in para_name:
                    para_name = para_name.replace("shortcut", "downsample")
                p.data.copy_(res18_state[para_name].data)

        self.freeze_backbone(freeze_bn = True)

    def load_ImageNet800_pretrained_model(self):
        assert self.header_mode == 'big', "Only standard ResNet18 architecture can load ImageNet pretrained model"
        res18 = torch.load(self.args['model']['selector']['pretrain_model'])
        res18 = res18['state_dict']
        prefix = "module.backbone."
        for name, p in self.named_parameters():
            if 'linear' not in name:
                p.data.copy_(res18[prefix+name].data.detach().cpu())

        self.freeze_backbone(freeze_bn = False)

    def freeze_feature(self):
        for name, p in self.named_parameters():
            if 'linear' not in name:
                p.requires_grad = False

    def unfreeze_feature(self):
        for name, p in self.named_parameters():
            if 'linear' not in name:
                p.requires_grad = True

    def get_classifier_params(self):
        for name, p in self.named_parameters():
            if 'linear' in name:
                yield p

    def freeze_backbone(self, freeze_bn = False):
        for name, p in self.named_parameters():
            if not('linear' in name):
                if freeze_bn:
                    p.requires_grad = False
                else:
                    if not ('bn' in name):
                        p.requires_grad = False

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        """
        Instantiates a ResNet layer.
        :param block: ResNet basic block
        :param planes: channels across the network
        :param num_blocks: number of blocks
        :param stride: stride
        :return: ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (output_classes)
        """
        out = self.header(x)
        out = self.layer1(out)  # 64, 32, 32
        out = self.layer2(out)  # 128, 16, 16
        out = self.layer3(out)  # 256, 8, 8
        out = self.layer4(out)  # 512, 4, 4
        out = avg_pool2d(out, out.shape[2]) # 512, 1, 1
        out = out.view(out.size(0), -1)  # 512
        out = self.linear(out)
        return out

    def predict(self, inputs=None, logits=None):
        out = self.forward(inputs)
        return out.argmax(dim=1)


def resnet18(args, nclasses: int, nf: int=64) -> ResNet:
    """
    Instantiates a ResNet18 network.
    :param nclasses: number of output classes
    :param nf: number of filters
    :return: ResNet network
    """
    header_mode = 'small'
    if 'header_mode' in args['model']['selector'] and args['model']['selector']['header_mode'] is not None:
        assert args['model']['selector']['header_mode'] in ['big', 'small']
        header_mode = args['model']['selector']['header_mode']
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, header_mode=header_mode, args=args)
