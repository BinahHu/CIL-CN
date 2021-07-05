import torch
import torch.nn as nn
from models.base import Base
from torch.optim import SGD as optimSGD
import backbone.ResNet as ResNet
import task_selectors.default as default


class Multi(Base):
    def __init__(self, args, transformer=None):
        super(Multi, self).__init__()
        self.args = args
        self.transformer = transformer
        self.mode = args['model']['multi_setting']
        self.task_num = args['dataset']['task_num']
        self.class_num = args['dataset']['class_num']
        self.class_per_task = self.class_num // self.task_num
        self.task_id = 0

        self.backbones = self.build_backbone()
        self.classifiers = self.build_classifier()
        self.selector = self.build_selector()
        self.opt = self.build_optim()
        self.loss = nn.CrossEntropyLoss()

    def build_optim(self, task_id=0):
        wd = 4e-3
        if 'weight_decay' in self.args['optim'] and self.args['optim']['weight_decay'] is not None:
            wd = self.args['optim']['weight_decay']
        if self.mode == 'full':
            return optimSGD([{'params': self.backbones[task_id].parameters()},
                             {'params': self.classifiers[task_id].parameters()}],
                            lr=self.args['optim']['lr'], weight_decay=wd, momentum=0.9)
        else:
            return optimSGD(self.classifiers[task_id].parameters(),
                            lr=self.args['optim']['lr'], weight_decay=wd, momentum=0.9)

    def build_backbone(self):
        if self.mode == 'full':
            res = []
            for i in range(self.task_num):
                res.append(ResNet.resnet18(self.args, 1))
            return nn.ModuleList(res)
        else:
            return nn.ModuleList([ResNet.resnet18(self.args, 1)])

    def build_classifier(self):
        res = []
        for i in range(self.task_num):
            res.append(nn.Linear(512, self.class_per_task))
        return nn.ModuleList(res)

    def build_selector(self):
        return default.DefaultSelector(self.args['dataset']['task_num'], self.args['dataset']['class_num'])

    def observe(self, inputs, labels, non_aug_input=None, logits=None):
        self.opt.zero_grad()
        if self.mode == 'full':
            outputs = self.classifiers[self.task_id](self.backbones[self.task_id].features(inputs))
        else:
            outputs = self.classifiers[self.task_id](self.backbones[0].features(inputs))
        labels = labels - self.task_id * self.class_per_task
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        pred = outputs.argmax(dim=1)
        correct = (pred == labels).sum().item()
        acc = correct / labels.shape[0]

        return loss.item(), 0, acc, 0

    def evaluate(self, inputs, labels):
        class_per_task = self.args['dataset']['class_num'] // self.args['dataset']['task_num']
        output_list = []
        for i in range(self.args['dataset']['task_num']):
            if self.mode == 'full':
                output_list.append(self.classifiers[i](self.backbones[i].features(inputs)))
            else:
                output_list.append(self.classifiers[i](self.backbones[0].features(inputs)))
        outputs = torch.cat(tuple(output_list), dim=1)
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

    def begin_task(self, args, t):
        self.lr = self.args['optim']['lr']
        self.task_id = t
        self.opt = self.build_optim(task_id=t)

    def begin_epoch(self, args, t, e):
        if args['optim']['drop'] is not None:
            if args['optim']['drop']['type'] == 'point':
                if e in args['optim']['drop']['val']:
                    self.lr = self.lr * 0.1
                    new_lr = self.lr
                    for i in range(len(self.opt.param_groups)):
                        self.opt.param_groups[i]['lr'] = new_lr

    def end_task(self, args, t):
        for name, p in self.classifiers[t].named_parameters():
            p.requires_grad = False
        if self.mode == 'full':
            for name, p in self.backbones[t].named_parameters():
                p.requires_grad = False