import torch
import torch.nn as nn
import copy
from inclearn.lib import factory

__all__ = ['multinet']


class MultiNet(nn.Module):
    def __init__(self,
                 sub_convnet=None,
                 sub_convnet_config={},
                 device=0):
        super(MultiNet, self).__init__()
        self.convnet_type = sub_convnet
        self.convnet_kwargs = sub_convnet_config

        self.convnets = nn.ModuleList()
        self.convnets.append(factory.get_convnet(self.convnet_type, **self.convnet_kwargs))
        self._out_dim = self.convnets[0].out_dim

        self.ntask = 0
        self.device = device

    def forward(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)

        return {"features": features}

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
            new_clf.load_state_dict(self.convnets[-1].state_dict())
            self.convnets.append(new_clf)



def multinet(**kwargs):
    """Constructs model which contains and maintains multiple convnets
    """
    model = MultiNet(**kwargs)
    return model



