import yaml
import utils.builder as builder
import argparse
import torch
import time
from datasets.imagenet import ImageNet
from torchvision import transforms
import os
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import shutil

def load_config(config_path):
    with open(config_path, 'r') as f:
        file_data = f.read()
        args = yaml.full_load(file_data)
    print("Arguments:")
    print(args)
    return args



def main():
    #Load config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    console_args = parser.parse_args()
    config_path = console_args.config
    args = load_config(config_path)

    # Build logger
    logger = builder.build_logger(args)

    #Build datasets and loader
    traindir = os.path.join(args['dataset']['root'], 'train')
    valdir = os.path.join(args['dataset']['root'], 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = ImageNet(
        traindir,
        train_transforms, mode='train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args['dataset']['batch_size'], shuffle=True,
                                               num_workers=48, pin_memory=True)
    train_loader_len = len(train_loader)

    val_loader = torch.utils.data.DataLoader(
        ImageNet(valdir, transforms.Compose([
            transforms.Resize(256),  # 256
            transforms.CenterCrop(224),  # 224
            transforms.ToTensor(),
            normalize,
        ]), mode='val'),
        batch_size=args['dataset']['batch_size'], shuffle=False, num_workers=48, pin_memory=True)
    val_loader_len = len(val_loader)
    args['dataset']['task_num'] = 1
    args['dataset']['class_num'] = 800

    #Build model
    model = builder.build_model(args, None)
    model.cuda()
    model = torch.nn.DataParallel(model).cuda()
    criterion = LabelSmoothingLoss(smoothing=0.1)

    #Build optimizer
    optimizer = torch.optim.SGD(model.parameters(), args['optim']['lr'],
                                momentum=0.9,
                                weight_decay=args['optim']['weight_decay'])

    #Load checkpoint
    start_epoch = 0
    best_prec = 0

    if args['ckpt']['resume']:
        ckpt = torch.load(os.path.join(args['ckpt']['root'], 'ckpt.pth.tar'))
        start_epoch = ckpt['epoch']
        best_prec = ckpt['best_prec1']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])

    cudnn.benchmark = True

    for e in range(start_epoch, args['optim']['epochs']):
        # train for one epoch
        train_loss, train_acc = train(train_loader, train_loader_len, model, criterion, optimizer, e, args)

        # evaluate on validation set
        val_loss, prec, prec5 = validate(val_loader, val_loader_len, model, criterion)

        lr = optimizer.param_groups[0]['lr']

        print("Epoch {:3d}, lr = {:4f}, train_loss={:6f}, val_loss={:6f}, train_acc={:2f}, val_acc={:2f}".format(e, lr, train_loss, val_loss, train_acc, prec))

        logger.logger.add_scalar('learning rate', lr, e + 1)
        logger.logger.add_scalars('loss', {'train loss': train_loss, 'validation loss': val_loss}, e + 1)
        logger.logger.add_scalars('accuracy', {'train accuracy': train_acc, 'validation accuracy': prec}, e + 1)

        is_best = prec > best_prec
        best_prec = max(prec, best_prec)
        save_checkpoint({
            'epoch': e + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=args['ckpt']['root'])


    logger.close()
    print('Best accuracy:')
    print(best_prec)


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(train_loader, train_loader_len, model, criterion, optimizer, epoch, args):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    mixup = 0.2
    end = time.time()
    if epoch < 100:
        mixup_alpha = mixup * float(epoch) / 100
    else:
        mixup_alpha = mixup
    for i, (input, target) in enumerate(train_loader):
        adjust_learning_rate(optimizer, epoch, i, train_loader_len, args)

        # measure data loading time
        data_time.update(time.time() - end)
        input = input.cuda()
        target = target.cuda()

        # compute output
        if mixup != 0:
            # using mixup
            input, label_a, label_b, lam = mixup_data(input, target, mixup_alpha)
            output = model(input)
            loss = mixup_criterion(criterion, output, label_a, label_b, lam)
            acc1_a, acc5_a = accuracy(output, label_a, topk=(1, 5))
            acc1_b, acc5_b = accuracy(output, label_b, topk=(1, 5))
            # measure accuracy and record loss
            prec1 = lam * acc1_a + (1 - lam) * acc1_b
            prec5 = lam * acc5_a + (1 - lam) * acc5_b
        else:
            # normal forward
            output = model(input)
            loss = criterion(output, target)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 100 == 0:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'LR: {4}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch+1, 120, i, train_loader_len, optimizer.param_groups[0]['lr'], batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
    return (losses.avg, top1.avg)

def validate(val_loader, val_loader_len, model, criterion):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 100 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, val_loader_len, batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return (losses.avg, top1.avg, top5.avg)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)


        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].float().sum()
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def adjust_learning_rate(optimizer, epoch, iteration, num_iter, args):

    warmup_epoch = 0
    warmup_iter = warmup_epoch * num_iter
    current_iter = iteration + epoch * num_iter


    count = sum([1 for s in args['optim']['drop']['val'] if s <= epoch])
    lr = args['optim']['lr'] * pow(args['optim']['gamma'], count)

    if epoch < warmup_epoch:
        lr = args['optim']['lr'] * current_iter / warmup_iter


    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def mixup_data(x, y, alpha):
    '''
    Returns mixed inputs, pairs of targets, and lambda
    '''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class LabelSmoothingLoss(nn.Module):

    def __init__(self, smoothing=0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = input.log_softmax(dim=-1)
        weight = input.new_ones(input.size()) * \
            self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='ckpt.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

if __name__ == '__main__':
    main()