import numpy as np
import torch



def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1

class Buffer:
    def __init__(self, args):
        self.task_num = args['dataset']['task_num']
        self.buffer_size = args['model']['buffer']['size']
        self.sub_buffer_size = self.buffer_size // self.task_num
        self.sub_dataset_size = args['dataset']['sub_set_size']
        self.task_id = 0
        self.device = args['device']
        self.mode = args['model']['buffer']['mode']
        self.num_seen_examples = 0


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
            self.labels = torch.zeros((self.buffer_size, *labels.shape[1:]), dtype=labels.dtype, device=self.device)
        if logits is not None:
            self.logits = torch.zeros((self.buffer_size, *logits.shape[1:]), dtype=logits.dtype, device=self.device)
        if task_labels is not None:
            self.task_labels = torch.zeros((self.buffer_size, *task_labels.shape[1:]), dtype=task_labels.dtype, device=self.device)
        self.init = True

    def gen_index(self):
        store_index = []
        for i in range(self.task_num):
            idx = np.arange(self.sub_dataset_size[i])
            np.random.shuffle(idx)
            store_index.append(idx[:self.sub_buffer_size])
        return store_index

    def is_empty(self):
        if self.mode == 'ring':
            return self.buffer_index == 0
        elif self.mode == 'reservoir':
            return self.num_seen_examples == 0

    def add_data(self, data, labels=None, logits = None, task_labels = None):
        if not self.init:
            self.init_buffer(data, labels, logits, task_labels)

        N = data.shape[0]
        t = self.task_id
        for i in range(N):
            if self.mode == 'ring':
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
            elif self.mode == 'reservoir':
                index = reservoir(self.num_seen_examples, self.buffer_size)
                self.num_seen_examples += 1
                if index >= 0:
                    self.data[index] = data[i]
                    if labels is not None:
                        self.labels[index] = labels[i]
                    if logits is not None:
                        self.logits[index] = logits[i]
                    if task_labels is not None:
                        self.task_labels[index] = task_labels[i]

    def get_data(self, size, transform=None):
        if self.mode == 'ring':
            if size > self.buffer_index:
                size = self.buffer_index
            choice = np.random.choice(self.buffer_index, size=size, replace=False)
        elif self.mode == 'reservoir':
            if size > min(self.num_seen_examples, self.data.shape[0]):
                size = min(self.num_seen_examples, self.data.shape[0])

            choice = np.random.choice(min(self.num_seen_examples, self.data.shape[0]),
                                      size=size, replace=False)
        if transform is None: transform = lambda x: x
        res = []
        res.append(torch.stack([transform(ee) for ee in self.data[choice]]).to(self.device))
        res.append(self.labels[choice] if self.labels is not None else None)
        res.append(self.logits[choice] if self.logits is not None else None)
        res.append(self.task_labels[choice] if self.task_labels is not None else None)
        return res

