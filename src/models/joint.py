import torch
import torch.nn as nn
from models.base import Base
from torch.optim import SGD as optimSGD
import backbone.ResNet as ResNet
import task_selectors.default as default



class Joint(Base):
    def __init__(self, args, transformer=None):
        super(Joint, self).__init__()
        self.args = args
        self.transformer = transformer

        self.backbone = self.build_backbone()
        self.selector = self.build_selector()
        self.opt = self.build_optim()
        self.loss = nn.CrossEntropyLoss()

    def build_optim(self):
        wd = 4e-3
        if 'weight_decay' in self.args['optim'] and self.args['optim']['weight_decay'] is not None:
            wd = self.args['optim']['weight_decay']
        return optimSGD(self.parameters(), lr=self.args['optim']['lr'], weight_decay=wd, momentum=0.9)

    def build_backbone(self):
        return ResNet.resnet18(self.args, self.args['dataset']['class_num'])

    def build_selector(self):
        return default.DefaultSelector(self.args['dataset']['task_num'], self.args['dataset']['class_num'])

    def observe(self, inputs, labels, non_aug_input=None, logits=None):
        self.opt.zero_grad()
        outputs = self.backbone(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        pred = outputs.argmax(dim=1)
        correct = (pred == labels).sum().item()
        acc = correct / labels.shape[0]

        return loss.item(), 0, acc, 0

    def evaluate(self, inputs, labels):
        class_per_task = self.args['dataset']['class_num'] // self.args['dataset']['task_num']
        outputs = self.backbone(inputs)
        TC_pred = self.selector(outputs)
        TIL_pred = torch.zeros(labels.shape).to(torch.device(self.args['device']))
        TC_correct = 0
        for i in range(outputs.shape[0]):
            gt_task_id = labels[i] // class_per_task
            task_id = TC_pred[i]
            TC_correct += (gt_task_id == task_id)
            TIL_pred[i] = outputs[i][gt_task_id * class_per_task : (gt_task_id + 1) * class_per_task].argmax() + gt_task_id * class_per_task
            for j in range(outputs.shape[1]):
                if j < task_id * class_per_task or j >= (task_id + 1) * class_per_task:
                    outputs[i][j] = -float('inf')
        CIL_pred = outputs.argmax(dim=1)
        TIL_correct = (TIL_pred == labels).sum().item()
        CIL_correct = (CIL_pred == labels).sum().item()

        return TIL_correct, CIL_correct, TC_correct