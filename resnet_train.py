'''
Code edited/taken from: github.com/akamaster
'''

import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import setup_resnet
import numpy as np

model_names = sorted(name for name in setup_resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(setup_resnet.__dict__[name]))

#print(model_names)

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', dest='start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
parser.add_argument('--device', dest='device', metavar='DEVICE', default='GPU', 
                    choices = ['CPU', 'GPU'],
                    help='Which device to use: CPU | GPU + (default: GPU)')
parser.add_argument('--threads', dest='threads', type=int, default=1,
                   help='how many threads to use, only relevant when device = CPU')
parser.add_argument('--dataset', dest='dataset', metavar='DATASET', default='CIFAR10', choices = ['CIFAR10', 'mixup', 'neighborhoodwatch'],
        help='Which dataset to use: CIFAR | mixup | neighborhoodwatch (default: CIFAR10)')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()


    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = torch.nn.DataParallel(setup_resnet.__dict__[args.arch]())
    if args.device == 'GPU':
        model.cuda()
    else:
        torch.set_num_threads(args.threads)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            print(checkpoint.keys())
            args.start_epoch = min(checkpoint['epoch'], args.start_epoch)
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    
    # Using torchvision normalize -> [-1, 1]
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    if args.dataset in ['CIFAR10', 'mixup']:
        train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
        batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True)
    elif args.dataset=='neighborhoodwatch' :
        datasetfname = "cvxCIFAR10dataset.pkl"
        with open(datasetfname, 'rb') as infile:
            neighbor_dataset  = torch.load(infile)
        train_loader = torch.utils.data.DataLoader(neighbor_dataset, 
                batch_size=args.batch_size, 
                shuffle=True, 
                num_workers=args.workers, 
                pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    if args.device == 'GPU':
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[100, 150], last_epoch=args.start_epoch - 1)

    if args.arch in ['resnet1202', 'resnet110']:
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this setup it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr*0.1


    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.th'))

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_dir, 'model.th'))


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    def nwatch_data(input, twotargets, weights):
        targets_a = twotargets[:,0]
        targets_b = twotargets[:,1]
        wa = weights[:,0]
        wb = weights[:,1]
        targets_a = targets_a.squeeze()
        targets_b = targets_b.squeeze()
        wa = wa.squeeze()
        wb = wb.squeeze()
        
        return input, targets_a, targets_b, wa, wb

    def mixup_data(input, targets, alpha=1.0):
        if alpha > 0.:
            lam = np.random.beta(alpha, alpha, input.size()[0])
        else:
            lam = np.ones(input.size()[0])
        
        batch_size = input.size()[0]
        index = torch.randperm(batch_size)
        lam = torch.Tensor(lam)
        lam = lam[:, None, None, None] # because inputs are 3-by-32-by-32 images
        if input.is_cuda:
            index = index.cuda()
            lam = lam.cuda()

        mixed_input = lam * input + (1 - lam) * input[index, :]
        target_a, target_b = targets, targets[index]
        return mixed_input, target_a, target_b, lam

        
    def nwatch_criterion(target_a, target_b, wa, wb):
        return lambda criterion, pred: (wa * criterion(pred, target_a) + wb * criterion(pred, target_b)).mean()
    
    def mixup_criterion(target_a, target_b, w):
        return lambda criterion, pred: (w * criterion(pred, target_a) + (1- w) * criterion(pred, target_b)).mean()

    end = time.time()
    for i, r in enumerate(train_loader):
        
        if args.dataset=='neighborhoodwatch':
            data_time.update(time.time() - end)
            
            input, twotargets, weights = r
            if args.device == 'GPU':
                input, twotargets, weights = input.cuda(), twotargets.cuda(), weights.cuda()
            
            # generated mixed inputs, two one-hot label vectors, and mixing coefficient
            inputs, targets_a, targets_b, wa, wb = nwatch_data(input, twotargets, weights)
            outputs = model(inputs)
            
            # compute output
            loss_func = nwatch_criterion(targets_a, targets_b, wa, wb)
            criterion = nn.CrossEntropyLoss(reduction='none')
            loss = loss_func(criterion, outputs)
            
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            outputs = outputs.float()
            loss = loss.float()
            
            _, predicted = torch.max(outputs.data, 1)
            correct = (wa * predicted.eq(targets_a.data)).cpu().sum() + (wb * predicted.eq(targets_b.data)).cpu().sum()
            prec1 = correct/input.size(0)
            
            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            
        elif args.dataset=='mixup':
            data_time.update(time.time() - end)

            input, targets = r
            if args.device == 'GPU':
                input, targets = input.cuda(), targets.cuda()

            # generate mixed inputs, two labels, and mixing coefficient
            inputs, targets_a, targets_b, w = mixup_data(input, targets)
            outputs = model(inputs)

            # compute output
            loss_func = mixup_criterion(targets_a, targets_b, w)
            criterion = nn.CrossEntropyLoss(reduction='none')
            loss = loss_func(criterion, outputs)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            outputs = outputs.float()
            loss = loss.float()

            _, predicted = torch.max(outputs.data, 1)
            correct = (w * predicted.eq(targets_a.data)).cpu().sum() + ( (1 - w) * predicted.eq(targets_b.data)).cpu().sum()
            prec1 = correct/input.size(0)

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

        else:
            input, target = r
            # measure data loading time
            data_time.update(time.time() - end)

            if args.device == 'GPU':
                target = target.cuda()
                input_var = input.cuda()
            else:
                input_var = input
            target_var = target
            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            output = output.float()
            loss = loss.float()
            
            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.device == 'GPU':
                target = target.cuda()
                input_var = input.cuda()
                target_var = target.cuda()
            else:
                input_var = input
                target_var = target

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
