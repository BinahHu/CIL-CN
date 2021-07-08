import torch
import torch.nn as nn
from models.base import Base
from torch.optim import SGD as optimSGD
import task_selectors.default as default
import task_selectors.ResNet as resnet
import task_selectors.ResNet_full as resnet_full
import utils.buffer as buffer
from torch.nn import functional as F


class Selector(Base):
    def __init__(self, args, transformer=None):
        super(Selector, self).__init__()
        self.args = args
        self.transform = transformer
        self.task_num = self.args['dataset']['task_num']
        self.class_num = self.args['dataset']['class_num']
        self.class_per_task = self.class_num // self.task_num
        self.backbone = self.build_backbone()
        self.selector = self.build_selector()
        self.buffer, self.buffer_loss = self.build_buffer()
        self.opt = self.build_optim()
        self.loss = nn.CrossEntropyLoss()
        self.task_id = 0

    def build_buffer(self):
        if self.args['model']['buffer'] is not None:
            return buffer.Buffer(self.args), self.args['model']['buffer']['loss']
        return None, None

    def build_selector(self):
        return default.DefaultSelector(self.args['dataset']['task_num'], self.args['dataset']['class_num'])

    def observe(self, inputs, labels, not_aug_inputs=None, logits=None):
        self.opt.zero_grad()
        outputs = self.backbone(inputs)
        labels = labels // self.class_per_task
        loss = self.loss(outputs, labels)

        pred = outputs.argmax(dim=1)
        correct = (pred == labels).sum().item()
        sample = labels.shape[0]

        if (self.buffer is not None) and (not self.buffer.is_empty()):
            buf_inputs, buf_labels, buf_logits, _ = self.buffer.get_data(
                self.args['model']['buffer']['batch_size'], transform=self.transform)
            buf_outputs = self.backbone(buf_inputs)
            loss += self.args['model']['buffer']['alpha'] * F.mse_loss(buf_outputs, buf_logits)

            buf_pred = buf_outputs.argmax(dim=1)
            correct += (buf_pred == buf_labels).sum().item()
            sample += buf_labels.shape[0]

            buf_inputs, buf_labels, _, _ = self.buffer.get_data(
                self.args['model']['buffer']['batch_size'], transform=self.transform)
            buf_outputs = self.backbone(buf_inputs)
            loss += self.args['model']['buffer']['beta'] * self.loss(buf_outputs, buf_labels)

            buf_pred = buf_outputs.argmax(dim=1)
            correct += (buf_pred == buf_labels).sum().item()
            sample += buf_labels.shape[0]

        loss.backward()
        self.opt.step()

        acc = correct / sample


        if self.buffer is not None:
            self.buffer.add_data(data=not_aug_inputs, labels=labels, logits=outputs.data)

        return loss.item(), 0, acc, 0

    def build_optim(self):
        return optimSGD(self.backbone.parameters(), lr=self.args['optim']['lr'])

    def build_backbone(self):
        return resnet.resnet18(args=self.args, nclasses=self.task_num)

    def evaluate(self, inputs, labels):
        class_per_task = self.args['dataset']['class_num'] // self.args['dataset']['task_num']
        TC_pred = self.backbone.predict(inputs=inputs, logits=None)
        TC_correct = 0
        for i in range(inputs.shape[0]):
            gt_task_id = labels[i] // class_per_task
            task_id = TC_pred[i]
            TC_correct += (gt_task_id == task_id)
        return 0, 0, TC_correct

    def begin_task(self, args, t):
        self.lr = self.args['optim']['lr']
        self.task_id = t
        self.buffer.set_status(t)

    def begin_epoch(self, args, t, e):
        if args['optim']['drop'] is not None:
            if args['optim']['drop']['type'] == 'point':
                if e in args['optim']['drop']['val']:
                    self.lr = self.lr * 0.1
                    new_lr = self.lr
                    for i in range(len(self.opt.param_groups)):
                        self.opt.param_groups[i]['lr'] = new_lr
