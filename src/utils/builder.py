import models
import datasets
from torch.utils.data import DataLoader


def build_dataset(args):
    dataset = datasets.datasets[args['dataset']['type']](args)
    return dataset, dataset.get_transform(), dataset.N_TASKS, dataset.N_CLASSES


def build_dataloader(args, dataset):
    train_loaders = []
    test_loaders = []
    task_num = args['dataset']['task_num']
    if args['model']['type'] == 'joint':
        for i in range(task_num - 1):
            train_loaders.append(None)
            test_loaders.append(None)
        train_loaders.append(DataLoader(dataset.sub_train_datasets[0], batch_size=args['dataset']['batch_size'],
                                        shuffle=True, num_workers=4))
        test_loaders.append(DataLoader(dataset.sub_test_datasets[0], batch_size=args['dataset']['batch_size'],
                                       shuffle=False, num_workers=4))
    else:
        for i in range(task_num):
            train_loaders.append(DataLoader(dataset.sub_train_datasets[i], batch_size=args['dataset']['batch_size'],
                                            shuffle=True, num_workers=4))
            test_loaders.append(DataLoader(dataset.sub_test_datasets[i], batch_size=args['dataset']['batch_size'],
                                           shuffle=False, num_workers=4))
    return train_loaders, test_loaders


def build_model(args, transformer):
    return models.models[args['model']['type']](args, transformer)


def build_logger(args):
    return None, None
