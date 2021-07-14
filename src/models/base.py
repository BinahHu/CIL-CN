import torch.nn as nn

class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()
        pass

    def build_optim(self):
        pass

    def build_backbone(self):
        pass

    def build_selector(self):
        pass

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

    def begin_rebalance(self, args, t):
        pass

    def end_rebalance(self, args, t):
        pass

    def begin_epoch_rebalance(self, args, t, e):
        pass

    def end_epoch_rebalance(self, args, t, e):
        pass