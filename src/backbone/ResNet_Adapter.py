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

class CovAdapter(nn.Module):
    expansion = 1

    def __init__(self, dim):
        super(CovAdapter, self).__init__()
        self.bn1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.conv1(out)

        out += residual
        out = self.bn2(out)

        return out

class BasicBlock(nn.Module):
    """
    The basic block of ResNet.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, task_num: int = 10, adapter = None) -> None:
        """
        Instantiates the basic block of the network.
        :param in_planes: the number of input channels
        :param planes: the number of channels (to be possibly expanded)
        :param task_num: total task num
        :param pca: use pca or not
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut_group = []
        if stride != 1 or in_planes != self.expansion * planes:
            for i in range(task_num):
                self.shortcut_group.append(nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                              stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                ))
        else:
            for i in range(task_num):
                self.shortcut_group.append(nn.Sequential())
        self.shortcut_group = nn.ModuleList(self.shortcut_group)
        self.task_num = task_num
        self.adapters1_group = []
        self.adapters2_group = []
        for i in range(task_num):
            self.adapters1_group.append(adapter(planes))
            self.adapters2_group.append(adapter(planes))
        self.adapters1_group = nn.ModuleList(self.adapters1_group)
        self.adapters2_group = nn.ModuleList(self.adapters2_group)
        self.task_id = 0

    def forward(self, x):
        """
        Compute a forward pass.
        :param x: list of input tensor (task_num, batch_size, input_size)
        :return: output tensor list, (task_num, 10)
        """
        i = self.task_id
        out = self.conv1(x)
        out = self.adapters1_group[i](out)
        out = relu(out)
        out = self.conv2(out)
        out = self.adapters2_group[i](out)
        out += self.shortcut_group[i](x)
        out = relu(out)
        return out

class Head(nn.Module):
    """
    CovNorm ResNet head
    """
    def __init__(self, nf: int, task_num: int, adapter, header_mode):
        super(Head, self).__init__()
        self.header_mode = header_mode

        self.adapters_group = []
        for i in range(task_num):
            self.adapters_group.append(adapter(nf * 1))
        self.adapters_group = nn.ModuleList(self.adapters_group)
        if self.header_mode == 'small':
            self.conv1 = conv3x3(3, nf * 1)
        else:
            self.conv1 = nn.Conv2d(3, nf * 1, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.task_num = task_num
        self.task_id = 0

    def forward(self, x):
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor list, (task_num, 10)
        """
        i = self.task_id
        if self.header_mode == 'small':
            return relu(self.adapters_group[i](self.conv1(x)))
        else:
            #x = F.interpolate(x, [256, 256], mode='bilinear')
            return self.maxpool(relu(self.adapters_group[i](self.conv1(x))))

class ResNet(nn.Module):
    """
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 task_num: int, class_per_task, nf, adapter=None, header_mode = 'small', args=None) -> None:
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
        self.task_num = task_num
        self.class_per_task = class_per_task
        self.class_num = task_num * class_per_task
        self.nf = nf
        self.header_mode = header_mode
        self.args = args
        self.head = Head(nf, task_num, adapter, header_mode)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1, task_num=task_num, adapter=adapter)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2, task_num=task_num, adapter=adapter)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2, task_num=task_num, adapter=adapter)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2, task_num=task_num, adapter=adapter)
        self.dropout = nn.Dropout(p=args['model']['dropout'])
        self.linear_group = []
        for i in range(task_num):
            self.linear_group.append(nn.Linear(nf * 8 * block.expansion, class_per_task))
        self.linear_group = nn.ModuleList(self.linear_group)

        if args is not None and 'pretrained' in args['model'] and args['model']['pretrained'] is not None:
            pretrained_source = args['model']['pretrained']
            if pretrained_source == 'ImageNet':
                self.load_ImageNet_pretrained_model()

    def load_ImageNet_pretrained_model(self):
        assert self.header_mode == 'big', "Only standard ResNet18 architecture can load ImageNet pretrained model"
        res18 = torchvision.models.resnet18(pretrained=True)

        x = list(res18.children())
        self.head.conv1.weight.data.copy_(x[0].weight.data)
        self.layer1[0].conv1.weight.data.copy_(x[4][0].conv1.weight.data)
        self.layer1[0].conv2.weight.data.copy_(x[4][0].conv2.weight.data)
        self.layer1[1].conv1.weight.data.copy_(x[4][1].conv1.weight.data)
        self.layer1[1].conv2.weight.data.copy_(x[4][1].conv2.weight.data)
        self.layer2[0].conv1.weight.data.copy_(x[5][0].conv1.weight.data)
        self.layer2[0].conv2.weight.data.copy_(x[5][0].conv2.weight.data)
        self.layer2[1].conv1.weight.data.copy_(x[5][1].conv1.weight.data)
        self.layer2[1].conv2.weight.data.copy_(x[5][1].conv2.weight.data)
        self.layer3[0].conv1.weight.data.copy_(x[6][0].conv1.weight.data)
        self.layer3[0].conv2.weight.data.copy_(x[6][0].conv2.weight.data)
        self.layer3[1].conv1.weight.data.copy_(x[6][1].conv1.weight.data)
        self.layer3[1].conv2.weight.data.copy_(x[6][1].conv2.weight.data)
        self.layer4[0].conv1.weight.data.copy_(x[7][0].conv1.weight.data)
        self.layer4[0].conv2.weight.data.copy_(x[7][0].conv2.weight.data)
        self.layer4[1].conv1.weight.data.copy_(x[7][1].conv1.weight.data)
        self.layer4[1].conv2.weight.data.copy_(x[7][1].conv2.weight.data)

        self.freeze_backbone()


    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int, task_num: int, adapter) -> nn.Module:
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
            layers.append(block(self.in_planes, planes, stride, task_num=task_num, adapter=adapter))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def set_status(self, task_id):
        self.task_id = task_id
        self.head.task_id = task_id
        for layers in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for layer in layers:
                layer.task_id = task_id

    def get_parameters(self, task_id, freeze_backbone=False):
        for name, p in self.named_parameters():
            if '_group' in name:
                if '_group.{}.'.format(task_id) in name:
                    yield p
            else:
                if task_id == 0 and (not freeze_backbone):
                    yield p

    def freeze_backbone(self):
        for name, p in self.named_parameters():
            if '_group' not in name:
                p.requires_grad = False

    def freeze_adapter(self, task_id):
        for name, p in self.named_parameters():
            if '_group.{}.'.format(task_id) in name:
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (output_classes)
        """
        outs = self.head(x)
        outs = self.layer1(outs)  # 64, 32, 32
        outs = self.layer2(outs)  # 128, 16, 16
        outs = self.layer3(outs)  # 256, 8, 8
        outs = self.layer4(outs)  # 512, 4, 4

        i = self.task_id
        out = avg_pool2d(outs, outs.shape[2])  # 512, 1, 1
        out = out.view(out.size(0), -1)  # 512
        out = self.dropout(out)
        out = self.linear_group[i](out)
        return out


def resnet18_adapter(args, task_num, class_per_task, nf: int=64) -> ResNet:
    """
    Instantiates a ResNet18 network.
    :param nclasses: number of output classes
    :param nf: number of filters
    :return: ResNet network
    """
    adapter = None
    if args['model']['type'] == 'covnorm':
        adapter = CovAdapter
    header_mode = 'small'
    if 'header_mode' in args['model'] and args['model']['header_mode'] is not None:
        assert args['model']['header_mode'] in ['big', 'small']
        header_mode = args['model']['header_mode']
    return ResNet(BasicBlock, [2, 2, 2, 2], task_num=task_num, class_per_task=class_per_task, nf=nf, adapter=adapter,
                  header_mode = header_mode, args=args)
