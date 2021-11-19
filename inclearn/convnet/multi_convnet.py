import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from inclearn.lib import factory
from collections import OrderedDict
import logging

__all__ = ['multinet']

logger = logging.getLogger(__name__)

class MultiNet(nn.Module):
    def __init__(self,
                 sub_convnet=None,
                 sub_convnet_config={},
                 pretrained_model=None,
                 prefix="",
                 with_super_split=False,
                 max_super_split=0,
                 use_oracle_super_label=True,
                 use_soft_mask=False,
                 use_bias=False,
                 reuse_old=False,
                 device=0):
        super(MultiNet, self).__init__()
        self.pretrained_model = pretrained_model
        self.prefix = prefix

        self.convnet_type = sub_convnet
        self.convnet_kwargs = sub_convnet_config

        self.convnets = nn.ModuleList()
        self.convnets.append(factory.get_convnet(self.convnet_type, **self.convnet_kwargs))
        if self.pretrained_model:
            self.load_pretrained_model(self.convnets[-1])
        self._out_dim = self.convnets[0].out_dim

        self.ntask = 0
        self.device = device
        self.with_super_split = with_super_split
        self.max_super_split = max_super_split
        self.use_oracle_super_label = use_oracle_super_label
        self.use_bias = use_bias
        self.reuse_old = reuse_old
        self.use_soft_mask = use_soft_mask

        if with_super_split and (not use_oracle_super_label):
            self.super_cat_classifier = nn.Linear(self._out_dim, self.max_super_split, bias=use_bias)
            if use_soft_mask:
                self.soft_mask_generator = nn.ModuleList()
                self.soft_mask_generator.append(nn.Linear(self.max_super_split, self._out_dim, bias=False))

    def load_pretrained_model(self, net):
        state_dict = torch.load(self.pretrained_model)
        for name, p in net.named_parameters():
            k = self.prefix + name
            if k in state_dict:
                if p.data.shape == state_dict[k].data.shape:
                    logger.info("Load parameter {}".format(name))
                    p.data.copy_(state_dict[k].data)
        for name, p in net.named_buffers():
            k = self.prefix + name
            if k in state_dict:
                logger.info("Load buffer parameter {}".format(name))
                p.data.copy_(state_dict[k].data)

    def gen_mask_with_id(self, y, feat_dim):
        assert feat_dim % self.max_super_split == 0
        mask = torch.zeros(y.shape[0], feat_dim).to(y.device)
        K = feat_dim // self.max_super_split
        for i in range(y.shape[0]):
            p = y[i]
            mask[i][p * K : (p+1) * K] = 1
        return mask

    def gen_mask_with_logits(self, score, feat_dim, ntask):
        assert feat_dim % self.max_super_split == 0
        mask = torch.ones(score.shape[0], feat_dim).to(score.device)
        K = feat_dim // self.max_super_split
        for i in range(score.shape[0]):
            for j in range(self.max_super_split):
                s = score[i][j]
                mask[i][j * K: (j + 1) * K] *= s
        return mask.repeat((1, ntask))


    def forward(self, x):
        if isinstance(x, list):
            inputs, mask_id = x
        else:
            inputs = x
            mask_id = None
        features = []
        super_cat_logits = None
        for i, convnet in enumerate(self.convnets):
            feat = convnet(inputs)["features"]
            if self.with_super_split and self.use_oracle_super_label and mask_id is not None:
                mask = self.gen_mask_with_id(mask_id, feat.shape[1])
                feat = feat * mask
            features.append(feat)
        features = torch.cat(features, 1)
        soft_mask = None
        if self.with_super_split and (not self.use_oracle_super_label):
            logits = self.super_cat_classifier(features)
            if self.use_soft_mask:
                full_mask = []
                for i in range(len(self.convnets)):
                    mask = self.soft_mask_generator[i](logits)
                    mask = F.sigmoid(mask)
                    full_mask.append(mask)
                full_mask = torch.cat(full_mask, 1)
                features = features * full_mask
                soft_mask = full_mask
            else:
                score = F.softmax(logits)
                mask = self.gen_mask_with_logits(score, self._out_dim, self.ntask)
                features = features * mask
            super_cat_logits = logits

        return {"features": features, "super_cat_logits": super_cat_logits, "soft_mask": soft_mask}

    @property
    def out_dim(self):
        return self._out_dim * len(self.convnets)

    @property
    def features_dim(self):
        return self._out_dim * len(self.convnets)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def copy(self):
        return copy.deepcopy(self)

    def add_classes(self, n_classes):
        self.ntask += 1

        if self.ntask > 1:
            new_clf = factory.get_convnet(self.convnet_type, **self.convnet_kwargs).to(self.device)
            if self.pretrained_model:
                self.load_pretrained_model(new_clf)
            else:
                new_clf.load_state_dict(self.convnets[-1].state_dict())
            self.convnets.append(new_clf)

            if self.with_super_split and (not self.use_oracle_super_label):
                new_super_clf = nn.Linear(self._out_dim * self.ntask, self.max_super_split, bias=self.use_bias).to(self.device)
                if self.use_soft_mask:
                    new_soft_generator = nn.Linear(self.max_super_split, self._out_dim, bias=False).to(self.device)
                    new_soft_generator.load_state_dict(self.soft_mask_generator[-1].state_dict())
                    self.soft_mask_generator.append(new_soft_generator)
                if self.reuse_old:
                    weight = copy.deepcopy(self.super_cat_classifier.weight.data)
                    new_super_clf.weight.data[:, :self._out_dim * (self.ntask - 1)] = weight
                    if self.use_bias:
                        bias = copy.deepcopy(self.super_cat_classifier.bias.data)
                        new_super_clf.bias = bias
                self.super_cat_classifier = new_super_clf

def multinet(**kwargs):
    """Constructs model which contains and maintains multiple convnets
    """
    model = MultiNet(**kwargs)
    return model



