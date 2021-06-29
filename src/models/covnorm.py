import torch
import torch.nn as nn
from models.base import Base
from torch.optim import SGD as optimSGD
import task_selectors.default as default
import task_selectors.ResNet as resnet
import utils.buffer as buffer
from torch.nn import functional as F

import backbone.ResNet_Adapter as ResNet_Adapter



class CovNorm(Base):
    def __init__(self, args, transformer=None):
        super(CovNorm, self).__init__()
        self.args = args
        self.transformer = transformer
        self.task_num = self.args['dataset']['task_num']
        self.class_num = self.args['dataset']['class_num']
        self.class_per_task = self.class_num // self.task_num
        self.backbone = self.build_backbone()
        self.selector = self.build_selector()
        self.buffer, self.buffer_loss = self.build_buffer()
        self.opt = self.build_optim()
        self.selector_opt = self.build_selector_optim()
        self.loss = nn.CrossEntropyLoss()
        self.backbone.set_status(0)

        self.task_id = 0

    def build_buffer(self):
        if self.args['model']['buffer'] is not None:
            return buffer.Buffer(self.args), self.args['model']['buffer']['loss']
        return None, None

    def build_selector(self):
        if 'backbone' in self.args['model']['selector'] and self.args['model']['selector']['backbone'] == 'ResNet18':
            return resnet.resnet18(args=self.args, nclasses=self.args['dataset']['task_num'])
        return default.DefaultSelector(self.args['dataset']['task_num'], self.args['dataset']['class_num'])

    def observe(self, inputs, labels, not_aug_inputs=None, logits=None):
        self.opt.zero_grad()
        outputs = self.backbone(inputs)
        task_labels = labels // self.class_per_task
        labels = labels - self.task_id * self.class_per_task
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        loss_task = torch.zeros(1).cuda()
        task_correct = 0
        task_sample = 0
        if (self.buffer is not None) and (not self.buffer.is_empty()):
            if self.buffer_loss == 'derpp':
                self.selector_opt.zero_grad()
                buf_inputs, buf_labels, buf_logits = self.buffer.get_data(
                    self.args['model']['buffer']['batch_size'], transform=self.transform)
                buf_outputs = self.selector(buf_inputs)
                loss_task += self.args['model']['buffer']['alpha'] * F.mse_loss(buf_outputs, buf_logits)

                buf_pred = buf_outputs.argmax(dim=1)
                task_correct += (buf_pred == buf_labels).sum().item()
                task_sample += buf_labels.shape[0]

                buf_inputs, buf_labels, _ = self.buffer.get_data(
                    self.args['model']['buffer']['batch_size'], transform=self.transform)
                buf_outputs = self.selector(buf_inputs)
                loss_task += self.args['model']['buffer']['beta'] * self.loss(buf_outputs, buf_labels)

                buf_pred = buf_outputs.argmax(dim=1)
                task_correct += (buf_pred == buf_labels).sum().item()
                task_sample += buf_labels.shape[0]

                loss_task.backward()
                self.selector_opt.step()

        buf_outputs = self.selector(inputs)

        pred = outputs.argmax(dim=1)
        correct = (pred == labels).sum().item()
        acc = correct / labels.shape[0]

        task_pred = buf_outputs.argmax(dim=1)
        task_correct += (task_pred == task_labels).sum().item()
        task_sample += task_labels.shape[0]
        task_acc = task_correct / task_sample

        if self.buffer is not None:
            self.buffer.add_data(examples=not_aug_inputs, labels=task_labels, logits=buf_outputs.data)

        return loss.item(), loss_task.item(), acc, task_acc

    def build_selector_optim(self):
        wd = 0
        if 'weight_decay' in self.args['model']['selector'] and self.args['model']['selector']['weight_decay'] is not None:
            wd = self.args['model']['selector']['weight_decay']
        return optimSGD(self.selector.parameters(), lr=self.args['model']['selector']['lr'], weight_decay=wd,
                        momentum=0.9)

    def build_optim(self, task_id=0):
        wd = 4e-3
        if 'weight_decay' in self.args['optim'] and self.args['optim']['weight_decay'] is not None:
            wd = self.args['optim']['weight_decay']
        return optimSGD(self.backbone.get_parameters(task_id), lr=self.args['optim']['lr'], weight_decay=wd, momentum=0.9)

    def build_backbone(self):
        return ResNet_Adapter.resnet18_adapter(args=self.args, task_num=self.task_num, class_per_task=self.class_per_task)

    def evaluate(self, inputs, labels):
        class_per_task = self.args['dataset']['class_num'] // self.args['dataset']['task_num']
        output_list = []
        for i in range(self.args['dataset']['task_num']):
            self.backbone.set_status(i)
            output_list.append(self.backbone(inputs))
        outputs = torch.cat(tuple(output_list), dim=1)
        TC_pred = self.selector.predict(outputs)
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
        self.selector_lr = self.args['model']['selector']['lr']
        if 'lr_adapter' in  self.args['optim'] and self.args['optim']['lr_adapter'] is not None and t > 0:
            self.lr = self.args['optim']['lr_adapter']
        self.task_id = t
        self.backbone.set_status(t)

        self.opt = self.build_optim(task_id=t)

    def begin_epoch(self, args, t, e):
        if args['optim']['drop'] is not None:
            if args['optim']['drop']['type'] == 'point':
                if e in args['optim']['drop']['val']:
                    self.lr = self.lr * 0.1
                    new_lr = self.lr
                    for i in range(len(self.opt.param_groups)):
                        self.opt.param_groups[i]['lr'] = new_lr
        if args['model']['selector']['drop'] is not None:
            if args['model']['selector']['drop']['type'] == 'point':
                if e in args['model']['selector']['drop']['val']:
                    self.selector_lr = self.selector_lr * 0.1
                    new_lr = self.selector_lr
                    for i in range(len(self.selector_opt.param_groups)):
                        self.selector_opt.param_groups[i]['lr'] = new_lr

    def end_task(self, args, t):
        self.backbone.freeze_adapter(t)
        if t == 0:
            self.backbone.freeze_backbone()