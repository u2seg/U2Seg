import argparse
import os
import random
import shutil
import time
import warnings
import pandas as pd
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import datasets as custom_datasets
# import torchvision.models as models

import resnet

# from lars import LARS
from vissl import optimizers
from data_preprocess import get_color_distortion, GaussianBlur, Clip

LARS = optimizers.lars._LARS

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

use_lars = True

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--trainindex_x', default=None, type=str, metavar='PATH',
                    help='path to train annotation_x (default: None)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=60, type=int, metavar='N',
                    help='number of total epochs to run')
if not use_lars:
    parser.add_argument('--decay-epochs', default="30,40,50", type=str, metavar='N,N,N',
                        help='number of epochs to run each decay (0.1x learning rate)')
parser.add_argument('--warmup-epochs', default=0, type=int, metavar='N',
                    help='number of epochs to warmup')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--freeze-backbone', action='store_true', default=False,
                    help='detach backbone (BN is updating)')
parser.add_argument('--skip-save', action='store_true', default=False,
                    help='skip checkpoint saving')
parser.add_argument('--no-lr-decay', action='store_true', default=False,
                    help='use no lr decay')
parser.add_argument('--num-classes', type=int, default=1000, choices=[100, 1000])
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)',
                    dest='weight_decay')
parser.add_argument('--nesterov', action='store_true', default=False,
                    help='use nesterov momentum')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrain', dest='pretrain', type=str, required=True,
                    help='use a pre-trained model to fine-tune')
parser.add_argument('--use-checkpoint', action='store_true',
                    help='use checkpointing to avoid out of memory')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--name-to-save', default='supervised', type=str,
                    help='name added to saved model file name')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--multi-epochs-data-loader', action='store_true', default=False,
                    help='use MultiEpochsDataLoader')
parser.add_argument('--use-persistent-workers', action='store_true', default=False,
                    help='use persistent workers')
parser.add_argument('--roll-data', default=None, type=int, help='roll data N times in the training dataset (does NOT change the number of epochs automatically)', metavar="N")
parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--no-syncbn', action='store_true', default=False,
                    help='do not use Sync BatchNorm')
parser.add_argument('--eval-freq', default=100, type=int,
                    metavar='N', help='evaluation epoch frequency (default: 100)')

best_acc1 = 0

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

        if False:
            cudnn.deterministic = True
            warnings.warn('You have chosen to seed training. '
                        'This will turn on the CUDNN deterministic setting, '
                        'which can slow down your training considerably! '
                        'You may see unexpected behavior when restarting '
                        'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    os.makedirs("saved", exist_ok=True)

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

class FineTuneModel(nn.Module):
    def __init__(self, state_dict, linear_dim=2048, num_classes=1000, head_first_layer_only=True, zero_init_logits_layer=False, freeze_backbone=False, **kwargs):
        super().__init__()

        model, head = resnet.get_resnet(**kwargs)
        
        self.model = model
        self.head = head

        self.model.load_state_dict(state_dict["resnet"])
        self.head.load_state_dict(state_dict["head"])

        if head_first_layer_only:
            self.head.layers = self.head.layers[:3]
        del model.fc

        self.linear = nn.Linear(linear_dim, num_classes)
        
        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # Set to true disturbs the gradient
        if zero_init_logits_layer:
            self.linear.weight.data.fill_(0.0)
            self.linear.bias.data.fill_(0.0)
    
    def forward(self, x):
        if False:
            # Keep fc layer in init if we want apply_fc=True
            x = self.model(x, apply_fc=True)
        else:
            x = self.model(x)
            if self.freeze_backbone:
                x = x.detach()
            # print(x)
            x = self.head(x)
            # print(x)
            x = self.linear(x)
            # print(x)
        return x

class LinearEvalModel(nn.Module):
    def __init__(self, state_dict, linear_dim=2048, num_classes=1000, head_first_layer_only=True, zero_init_logits_layer=False, **kwargs):
        super().__init__(**kwargs)

        model, head = resnet.get_resnet(**kwargs)
        
        self.model = model
        self.head = head

        self.model.load_state_dict(state_dict["resnet"])
        self.head.load_state_dict(state_dict["head"])

        if head_first_layer_only:
            self.head.layers = self.head.layers[:3]

        self.linear = nn.Linear(linear_dim, num_classes)
        
        # Set to true disturbs the gradient
        if zero_init_logits_layer:
            self.linear.weight.data.fill_(0.0)
            self.linear.bias.data.fill_(0.0)
    
    def forward(self, x):
        # Keep fc layer in init if we want apply_fc=True
        x = self.model(x, apply_fc=True)

        return x

# FineTuneModel = LinearEvalModel

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    
    state_dict = torch.load(args.pretrain, map_location="cpu")
    depth, width_multiplier, sk_ratio = resnet.name_to_params(args.pretrain)
    print("Using ImageNet", args.num_classes)
    model = FineTuneModel(state_dict, depth=depth, width_multiplier=width_multiplier, sk_ratio=sk_ratio, checkpointing=args.use_checkpoint, linear_dim=2048 * width_multiplier, 
        num_classes=args.num_classes, freeze_backbone=args.freeze_backbone)

    if False:
        print("Head:")
        print(model.head)
        print("Linear:")
        print(model.linear)

    if args.gpu is not None:
        if args.no_syncbn:
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
                print("Not converting SyncBN")
        else:
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
                print("Enable SyncBN conversion")
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)

    # define optimizer
    if use_lars:
        # Note: we don't have weight decay in fine tuning. exclude_from_weight_decay is not effective and not implemented.
        optimizer = LARS(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            exclude_bias_and_norm=True,
        )

        # "decay the learning rate with the cosine decay schedule without restarts"
        if not args.no_lr_decay:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, args.epochs, eta_min=0, last_epoch=-1
            )
        else:
            scheduler = None
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=args.nesterov)

    if args.amp_opt_level != "O0":
        if amp is None:
            print("apex is not installed but amp_opt_level is set to {args.amp_opt_level}, ignoring.\n"
                           "you should install apex from https://github.com/NVIDIA/apex#quick-start first")
            args.amp_opt_level = "O0"
        else:
            model, optimizer = amp.initialize(model.cuda(), optimizer, opt_level=args.amp_opt_level)

    assert torch.cuda.is_available(), "CUDA is not available"
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                if isinstance(best_acc1, torch.Tensor):
                    best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if args.amp_opt_level != "O0" and checkpoint['args'].amp_opt_level != "O0":
                amp.load_state_dict(checkpoint['amp'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    # "~" will be expanded to absolute path
    if not args.trainindex_x.startswith("/") and not args.trainindex_x.startswith("./"):
        index_info_x = os.path.join(args.data, 'indexes', args.trainindex_x)
    else:
        index_info_x = args.trainindex_x

    if index_info_x.endswith(".npy"):
        trainindex_x = np.load(index_info_x)
        index_info_x = {'Index': trainindex_x}
        trainindex_x = index_info_x['Index'].tolist()
    else: # Expect csv files
        index_info_x = pd.read_csv(index_info_x)
        trainindex_x = index_info_x['Index'].tolist()

    train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    if 'Pseudolabel' in index_info_x:
        print("Using pseudo-labels")
        labels_x = index_info_x['Pseudolabel'].tolist()
        train_dataset = custom_datasets.ImageFolderWithIndexAndTarget(
            traindir, trainindex_x, target=labels_x, transform=train_transforms)
        # print(labels_x)
    else:
        print("Not using pseudo-labels")
        train_dataset = custom_datasets.ImageFolderWithIndex(
            traindir, trainindex_x, transform=train_transforms)    

    if args.roll_data is not None:
        assert isinstance(train_dataset.samples, list)
        assert isinstance(train_dataset.targets, list)
        assert isinstance(train_dataset.imgs, list)
        assert isinstance(args.roll_data, int)
        roll_data = args.roll_data
        train_dataset.samples = train_dataset.samples * roll_data
        train_dataset.targets = train_dataset.targets * roll_data
        train_dataset.imgs = train_dataset.imgs * roll_data

        print("Dataset length:", len(train_dataset))


    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    if args.multi_epochs_data_loader:
        from dataloader import MultiEpochsDataLoader
        DataLoader = MultiEpochsDataLoader
    else:
        DataLoader = torch.utils.data.DataLoader

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, persistent_workers=args.use_persistent_workers)

    val_loader = DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # This is with SGD optimizer
        if not use_lars:
            adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        if scheduler is not None:
            scheduler.step()

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            print("LR: ", optimizer.param_groups[0]["lr"])

            is_best = False
            if (epoch + 1) % args.eval_freq == 0:
                # evaluate on validation set
                acc1 = validate(val_loader, model, criterion, args)
                # remember best acc@1 and save checkpoint
                is_best = acc1 > best_acc1
                best_acc1 = max(acc1, best_acc1)
            
            save_checkpoint(args, {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'args': args,
                'amp':  amp.state_dict() if args.amp_opt_level != 'O0' else None
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if args.amp_opt_level != "O0":
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        
        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print(name)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.rank == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and args.rank == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(args, state, is_best, filename=None):
    if args.skip_save:
        return
    
    if filename is None:
        filename = 'saved/checkpoint_{}.pth.tar'.format(args.name_to_save)
    torch.save(state, filename)
    # if is_best:
    #     shutil.copyfile(filename, 'model_best_{}.pth.tar'.format(args.name_to_save))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every `decay_epochs` epochs"""
    if epoch < args.warmup_epochs:
        lr = args.lr * ((epoch + 1) / args.warmup_epochs)
    else:
        decay_eps = np.array([int(i) for i in args.decay_epochs.split(",")])
        decay_times = np.count_nonzero(epoch > decay_eps)
        lr = args.lr * (0.1 ** decay_times)
    print("[Epoch {}]Adjust learning rate to: {}".format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
