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

def eval(args, model, val_loader, t, device):
    model.eval()
    model.backbone.eval()
    model.selector.eval()
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

def main():
    #Load config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    console_args = parser.parse_args()
    config_path = console_args.config
    args = load_config(config_path)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args['device'] = device

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

    #Build model and load ckpt
    model = builder.build_model(args, transformer)
    model.to(torch.device(device))
    if 'backbone' in args['ckpt']:
        model.backbone.load_state_dict(torch.load(args['ckpt']['backbone']))
    if 'selector' in args['ckpt']:
        model.selector.load_state_dict(torch.load(args['ckpt']['selector']))

    for name, p in model.named_parameters():
        p.requires_grad = False

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