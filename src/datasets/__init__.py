import importlib


file_class_pair = {'cifar10':'SeqCIFAR10',
                   'cifar100':'SeqCIFAR100',
                   'tinyimg':'SeqTinyImg'}

datasets = {}
for k,v in file_class_pair.items():
    mod = importlib.import_module('datasets.' + k)
    datasets[k] = getattr(mod, v)