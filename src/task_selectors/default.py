import torch
import torch.nn as nn



class DefaultSelector(nn.Module):
    def __init__(self, task_num, class_num):
        super(DefaultSelector, self).__init__()
        self.task_num = task_num
        self.class_num = class_num

    def forward(self, x):
         max_unit = x.argmax(dim=1)
         class_per_task = self.class_num // self.task_num
         max_task = max_unit // class_per_task
         return max_task