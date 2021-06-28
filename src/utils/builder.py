import models
import datasets
from torch.utils.data import DataLoader
from utils.logger import Logger


def build_dataset(args):
    dataset = datasets.datasets[args['dataset']['type']](args)
    return dataset, dataset.get_transform(), dataset.N_TASKS, dataset.N_CLASSES


def build_dataloader(args, dataset):
    train_loaders = []
    test_loaders = []
    task_num = args['dataset']['task_num']
    for i in range(task_num):
        train_loaders.append(DataLoader(dataset.sub_train_datasets[i], batch_size=args['dataset']['batch_size'],
                                        shuffle=True, num_workers=48))
        test_loaders.append(DataLoader(dataset.sub_test_datasets[i], batch_size=args['dataset']['batch_size'],
                                       shuffle=False, num_workers=48))
    return train_loaders, test_loaders


def build_model(args, transformer):
    return models.models[args['model']['type']](args, transformer)


def build_logger(args):
    if args['logger'] is None:
        return None

    return Logger(args)
