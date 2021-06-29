import numpy as np
import torch


class Buffer:
    def __init__(self, args):
        self.task_num = args['dataset']['task_num']
        self.buffer_size = args['model']['buffer']['size']
        self.sub_buffer_size = self.buffer_size // self.task_num
        self.sub_dataset_size = args['dataset']['sub_set_size']
        self.task_id = 0

        self.store_index = self.gen_index()
        self.task_index = [0] * self.task_num
        self.buffer_index = 0
        self.init = False
        self.data = None
        self.labels = None
        self.logits = None
        self.task_labels = None

    def set_status(self, task_id):
        self.task_id = task_id

    def init_buffer(self, data, labels, logits, task_labels):
        self.data = torch.zeros((self.buffer_size, *data.shape[1:]), dtype=data.dtype)
        if labels is not None:
            self.labels = torch.zeros((self.buffer_size, *labels.shape[1:]), dtype=labels.dtype)
        if logits is not None:
            self.logits = torch.zeros((self.buffer_size, *logits.shape[1:]), dtype=logits.dtype)
        if task_labels is not None:
            self.task_labels = torch.zeros((self.buffer_size, *task_labels.shape[1:]), dtype=task_labels.dtype)
        self.init = True

    def gen_index(self):
        store_index = []
        for i in range(self.task_num):
            idx = np.arange(self.sub_dataset_size[i])
            np.random.shuffle(idx)
            store_index.append(idx[:self.sub_buffer_size])
        return store_index

    def is_empty(self):
        return self.buffer_index == 0

    def add_data(self, data, labels=None, logits = None, task_labels = None):
        if not self.init:
            self.init_buffer(data, labels, logits, task_labels)

        N = data.shape[0]
        t = self.task_id
        for i in range(N):
            if self.task_index[t] in self.store_index[t]:
                self.data[self.buffer_index] = data[i]
                if labels is not None:
                    self.labels[self.buffer_index] = labels[i]
                if logits is not None:
                    self.logits[self.buffer_index] = logits[i]
                if task_labels is not None:
                    self.task_labels[self.buffer_index] = task_labels[i]
                self.buffer_index += 1
            self.task_index[t] += 1

    def get_data(self, size, transform=None):
        if size > self.buffer_index:
            size = self.buffer_index
        choice = np.random.choice(self.buffer_index, size=size, replace=False)
        if transform is None: transform = lambda x: x
        res = []
        res.append(torch.stack([transform(ee) for ee in self.data[choice]]))
        res.append(self.labels[choice] if self.labels is not None else None)
        res.append(self.logits[choice] if self.logits is not None else None)
        res.append(self.task_labels[choice] if self.task_labels is not None else None)
        return res

