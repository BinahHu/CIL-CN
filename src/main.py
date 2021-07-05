import yaml
import utils.builder as builder
import argparse
import torch
from utils.logger import progress_bar

def load_config(config_path):
    with open(config_path, 'r') as f:
        file_data = f.read()
        args = yaml.full_load(file_data)
    print("Arguments:")
    print(args)
    return args

def eval(args, model, val_loader, t, device):
    model.eval()
    TIL_correct = 0
    CIL_correct = 0
    TC_correct = 0
    total = 0
    for data in val_loader:
        img, target = data
        img = img.to(device)
        target = target.to(device)
        TIL_correct_batch, CIL_correct_batch, TC_correct_batch = model.evaluate(img, target)
        TIL_correct += TIL_correct_batch
        CIL_correct += CIL_correct_batch
        TC_correct += TC_correct_batch
        total += target.shape[0]

    TIL_acc = TIL_correct / total * 100
    CIL_acc = CIL_correct / total * 100
    TC_acc = TC_correct / total * 100

    return TIL_acc, CIL_acc, TC_acc, total


def train(args, model, train_loader, logger, t, device):
    model.train()
    model.begin_task(args, t)
    loss_acc = 0
    loss_task_acc = 0
    acc_acc = 0
    acc_task_acc = 0
    cnt = 0
    iter_num = 0
    for e in range(args['optim']['epochs']):
        model.begin_epoch(args, t, e)
        for i, data in enumerate(train_loader):
            logits = None
            if hasattr(train_loader.dataset, 'logits'):
                inputs, labels, not_aug_inputs, logits = data
            else:
                inputs, labels, not_aug_inputs = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            loss, loss_task, acc, acc_task = model.observe(inputs, labels, not_aug_inputs, logits)

            loss_acc += loss
            loss_task_acc += loss_task
            acc_acc += acc * 100
            acc_task_acc += acc_task * 100
            cnt += 1
            progress_bar(i, len(train_loader), e, t,
                         loss_acc / cnt, loss_task_acc / cnt,
                         acc_acc / cnt, acc_task_acc / cnt)

            iter_num += 1
            logger.log(t, iter_num, loss, loss_task, acc, acc_task)

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

    # Build logger
    logger = builder.build_logger(args)

    #Build datasets
    dataset, transformer, task_num, class_num = builder.build_dataset(args)
    args['dataset']['task_num'] = task_num
    args['dataset']['class_num'] = class_num

    #Buil dataloader
    train_loaders, val_loaders = builder.build_dataloader(args, dataset)
    sub_set_size = []
    for t in range(task_num):
        if train_loaders[t] is None:
            sub_set_size.append(0)
        else:
            sub_set_size.append(len(train_loaders[t]))
    args['dataset']['sub_set_size'] = sub_set_size

    #Build model
    model = builder.build_model(args, transformer)
    model.to(torch.device(device))


    #Training loop
    for t in range(task_num):
        if args['model']['type'] == 'joint' and t < args['dataset']['task_num'] - 1:
            continue
        train_loader = train_loaders[t]
        train(args, model, train_loader, logger, t, args['device'])
        print()
    # Evaluation loop
    TIL = []
    CIL = []
    TC = []
    N = 0
    for t in range(task_num):
        val_loader = val_loaders[t]
        TIL_acc, CIL_acc, TC_acc, sample_num = eval(args, model, val_loader, t, args['device'])
        TIL.append(TIL_acc * sample_num)
        CIL.append(CIL_acc * sample_num)
        TC.append(TC_acc * sample_num)
        N += sample_num
        print("Task {}, TIL acc = {:.2f}, CIL acc = {:.2f}".format(t, TIL_acc, CIL_acc))

    print("Average TIL acc = {:.2f}, average CIL acc = {:.2f}, average TC acc = {:.2f}".format(sum(TIL) / N, sum(CIL) / N, sum(TC) / N))

if __name__ == '__main__':
    main()