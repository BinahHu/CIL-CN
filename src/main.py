import yaml
import utils.builder as builder
import argparse
import torch

def load_config(config_path):
    with open(config_path, 'r') as f:
        file_data = f.read()
        args = yaml.full_load(file_data)
    print("Arguments:")
    print(args)
    return args

def eval(args, model, val_loader, csv_logger, tb_logger, t, device):
    pass

def train(args, model, train_loader, csv_logger, tb_logger, t, device):
    model.begin_task(args, t)
    for e in range(args['optim']['epoch']):
        model.begin_epoch(args, t, e)

        model.end_epoch(args, t, e)
    model.end_task(args, t)

def main():
    #Load config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    console_args = parser.parse_args()
    config_path = console_args.config
    args = load_config(config_path)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args['device'] = device

    #Build dataset
    dataset, transformer, task_num, class_num = builder.build_dataset(args)
    args['dataset']['task_num'] = task_num
    args['dataset']['class_num'] = class_num

    #Buil dataloader
    train_loaders, val_loaders = builder.build_dataloader(args)

    #Build model
    model = builder.build_model(args, transformer)
    model.to(torch.device(device))

    #Build logger
    csv_logger, tb_logger = builder.build_logger(args)

    #Training loop
    for t in range(task_num):
        train_loader = train_loaders[t]
        val_loader = val_loaders[t]
        train(args, model, train_loader, csv_logger, tb_logger, t)
        eval(args, model, val_loader, csv_logger, tb_logger, t)

if __name__ == '__main__':
    main()