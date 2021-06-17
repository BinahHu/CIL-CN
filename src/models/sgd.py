import torch
import torch.nn as nn
from models.base import Base


class SGD(Base):
    def __init__(self, args, transformer=None):
        super(SGD, self).__init__(args, transformer=None)
        self.loss = nn.CrossEntropyLoss()

    def observe(self, inputs, labels, non_aug_input=None, logits=None):
        self.opt.zero_grad()
        outputs = self.backbone(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        return loss.item()

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