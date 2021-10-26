"""Taken & slightly modified from:
* https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from torch.nn.functional import relu, avg_pool2d
import torchvision
import re

__all__ = ['ResNet', 'resnet18_adapter']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

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

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """
    The basic block of ResNet.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, adapter = None) -> None:
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

        self.shortcut_group = nn.ModuleList([])
        self.adapters1_group = nn.ModuleList([])
        self.adapters2_group = nn.ModuleList([])
        self.task_id = 0

        self.stride = stride
        self.in_planes = in_planes
        self.planes = planes
        self.adapter = adapter

    def add_classes(self, n_classes, device=0):
        if self.stride != 1 or self.in_planes != self.expansion * self.planes:
            self.shortcut_group.append(nn.Sequential(
                nn.Conv2d(self.in_planes, self.expansion * self.planes, kernel_size=1,
                          stride=self.stride, bias=False),
                nn.BatchNorm2d(self.expansion * self.planes)
            ).to(device))
        else:
            self.shortcut_group.append(nn.Sequential().to(device))
        self.adapters1_group.append(self.adapter(self.planes).to(device))
        self.adapters2_group.append(self.adapter(self.planes).to(device))

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
    def __init__(self, nf: int, adapter, header_mode):
        super(Head, self).__init__()
        self.header_mode = header_mode

        self.adapters_group = nn.ModuleList([])
        if self.header_mode == 'small':
            self.conv1 = conv3x3(3, nf * 1)
        else:
            self.conv1 = nn.Conv2d(3, nf * 1, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.task_id = 0

        self.adapter = adapter
        self.nf = nf

    def add_classes(self, n_classes, device=0):
        self.adapters_group.append(self.adapter(self.nf * 1).to(device))

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
            x = F.interpolate(x, [256, 256], mode='bilinear')
            return self.maxpool(relu(self.adapters_group[i](self.conv1(x))))

class ResNet(nn.Module):
    """
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(
            self,
            block,
            layers,
            nf=64,
            header_mode='small',
            adapter=CovAdapter,
            pretrained=False,
            pretrained_model=None,
            pretrained_mode_path=None,
            device=0,
            **kwargs
    ):

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
        self.nf = nf
        self.header_mode = header_mode
        self.head = Head(nf, adapter, header_mode)
        self.layer1 = self._make_layer(block, nf * 1, layers[0], stride=1, adapter=adapter)
        self.layer2 = self._make_layer(block, nf * 2, layers[1], stride=2, adapter=adapter)
        self.layer3 = self._make_layer(block, nf * 4, layers[2], stride=2, adapter=adapter)
        self.layer4 = self._make_layer(block, nf * 8, layers[3], stride=2, adapter=adapter)

        self._out_dim = nf * 8 * block.expansion

        self.task_id = -1
        self.is_train = True
        self.pretrained = pretrained
        self.device = device

        if pretrained:
            if pretrained_model == 'ImageNet':
                self.load_ImageNet_pretrained_model()
            elif pretrained_model == 'ImageNet-800':
                self.load_ImageNet800_pretrained_model(pretrained_mode_path)

    @property
    def out_dim(self):
        return self._out_dim

    def add_classes(self, n_classes):
        self.set_status(self.task_id + 1, True)
        return
        self.head.add_classes(n_classes, device=self.device)
        for layers in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for layer in layers:
                layer.add_classes(n_classes, device=self.device)

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

    def load_ImageNet800_pretrained_model(self, pretrained_model):
        for i in range(4):
            self.head.add_classes(0, device=self.device)
            for layers in [self.layer1, self.layer2, self.layer3, self.layer4]:
                for layer in layers:
                    layer.add_classes(0, device=self.device)
        assert self.header_mode == 'big', "Only standard ResNet18 architecture can load ImageNet pretrained model"
        res18 = torch.load(pretrained_model)
        res18 = res18['state_dict']
        prefix = "module.backbone."
        for name, p in self.named_parameters():
            if '_group' in name:
                if 'linear_group' not in name:
                    task_id = int(re.match(r'.*group\.([0-9]*)\..*', name).group(1))
                    p.data.copy_(res18[prefix+name.replace("_group.{}.".format(task_id), "_group.0.")].detach().cpu())
            else:
                p.data.copy_(res18[prefix+name].data.detach().cpu())

        self.freeze_backbone()


    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int, adapter) -> nn.Module:
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
            layers.append(block(self.in_planes, planes, stride, adapter=adapter))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def set_status(self, task_id, train=True):
        self.task_id = task_id
        self.is_train = train
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

    def forward(self, x: torch.Tensor):
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (output_classes)
        """
        if self.is_train:
            outs = self.head(x)
            outs = self.layer1(outs)  # 64, 32, 32
            outs = self.layer2(outs)  # 128, 16, 16
            outs = self.layer3(outs)  # 256, 8, 8
            outs = self.layer4(outs)  # 512, 4, 4

            out = avg_pool2d(outs, outs.shape[2])  # 512, 1, 1
            out = out.view(out.size(0), -1)  # 512

            return {"features": out}
        else:
            max_task = self.task_id
            out_list = []
            for i in range(max_task + 1):
                self.set_status(i, False)
                outs = self.head(x)
                outs = self.layer1(outs)  # 64, 32, 32
                outs = self.layer2(outs)  # 128, 16, 16
                outs = self.layer3(outs)  # 256, 8, 8
                outs = self.layer4(outs)  # 512, 4, 4

                out = avg_pool2d(outs, outs.shape[2])  # 512, 1, 1
                out = out.view(out.size(0), -1)  # 512
                out_list.append(out)
            return {"features": out_list}

def resnet18_adapter(pretrained=False,
                     pretrained_model=None,
                     pretrained_mode_path=None,
                     header_mode = 'big',
                     **kwargs):
    """
    Instantiates a ResNet18 network.
    :param nclasses: number of output classes
    :param nf: number of filters
    :return: ResNet network
    """
    return ResNet(BasicBlock, [2, 2, 2, 2],
                  header_mode = header_mode,
                  adapter=CovAdapter,
                  pretrained=pretrained,
                  pretrained_model=pretrained_model,
                  pretrained_mode_path=pretrained_mode_path,
                  **kwargs)