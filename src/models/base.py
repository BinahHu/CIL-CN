import torch
import torch.nn as nn
from torch.optim import SGD
import backbone.ResNet as ResNet
import task_selectors.default as default


class Base(nn.Module):
    def __init__(self, args, transformer=None):
        super(Base, self).__init__()

        self.args = args
        self.transformer = transformer

        self.backbone = self.build_backbone()
        self.selector = self.build_selector()
        self.opt = self.build_optim()

    def build_optim(self):
        return SGD(self.parameters(), lr=self.args['optim']['lr'])

    def build_backbone(self):
        return ResNet.resnet18(self.args['dataset']['class_num'])

    def build_selector(self):
        return default.DefaultSelector(self.args['dataset']['task_num'], self.args['dataset']['class_num'])

    def observe(self, inputs, labels, non_aug_input=None, logits=None):
        pass

    def evaluate(self, inputs, labels):
        pass

    def begin_task(self, args, t):
        pass

    def end_task(self, args, t):
        pass

    def begin_epoch(self, args, t, e):
        pass

    def end_epoch(self, args, t, e):
        pass