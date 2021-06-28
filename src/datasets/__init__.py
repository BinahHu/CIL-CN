import importlib


file_class_pair = {'cifar10':'SeqCIFAR10',
                   'tinyimg':'SeqTinyImg',
                   'imagenet':'ImageNet',
                   'cifar100':'SeqCIFAR100'}

datasets = {}
for k,v in file_class_pair.items():
    mod = importlib.import_module('datasets.' + k)
    datasets[k] = getattr(mod, v)