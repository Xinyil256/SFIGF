import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils_ori import *
from datasets import *
from models import *
# from models.option import args as dagf_args



# print(dagf_args)

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='jiif')
parser.add_argument('--model', type=str, default='JIIF')
parser.add_argument('--loss', type=str, default='L1')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dataset', type=str, default='NYU')
parser.add_argument('--data_root', type=str, default='/home/amax/Documents/yxxxl/oridata/NYUDepthv2/NYUDepthv2/')
parser.add_argument('--train_batch', type=int, default=1)
parser.add_argument('--test_batch', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--epoch', default=100, type=int, help='max epoch')
parser.add_argument('--eval_interval',  default=10, type=int, help='eval interval')
parser.add_argument('--checkpoint',  default='scratch', type=str, help='checkpoint to use')
parser.add_argument('--scale',  default=8, type=int, help='scale')
parser.add_argument('--interpolation',  default='bicubic', type=str, help='interpolation method to generate lr depth')
parser.add_argument('--lr',  default=0.0001, type=float, help='learning rate')
parser.add_argument('--lr_step',  default=40, type=float, help='learning rate decay step')
parser.add_argument('--lr_gamma',  default=0.2, type=float, help='learning rate decay gamma')
parser.add_argument('--input_size',  default=None, type=int, help='crop size for hr image')
parser.add_argument('--sample_q',  default=30720, type=int, help='sampled pixels per hr depth')
parser.add_argument('--noisy',  action='store_true', help='add noise to train dataset')
parser.add_argument('--test',  action='store_true', help='test mode')
parser.add_argument('--report_per_image',  action='store_true', help='report RMSE of each image')
parser.add_argument('--save',  action='store_true', help='save results')
parser.add_argument('--batched_eval',  action='store_true', help='batched evaluation to avoid OOM for large image resolution')
parser.add_argument('--base_channel',  default=64, type=int, help='batched evaluation to avoid OOM for large image resolution')
parser.add_argument('--in_ch',  default=64, type=int, help='in_ch of swinir')
parser.add_argument('--out_ch',  default=64, type=int, help='out_ch of swinir')
parser.add_argument('--no_pre_upsample', action='store_false', help='disable data_preupsample')
parser.add_argument('--num_features', type=int, default=32)
parser.add_argument('--in_channels', type=int, default=1)
parser.add_argument('--guide_channels', type=int, default=3)
parser.add_argument('--num_pyramid', type=int, default=3)
parser.add_argument('--act', type=str, default='PReLU')
parser.add_argument('--norm', type=str, default='None')
parser.add_argument('--filter_size', type=int, default=3)  # 生成的kernel的大小
parser.add_argument('--num_res', type=int, default=2)
parser.add_argument('--transformation', type=bool, default=False)
args = parser.parse_args()

seed_everything(args.seed)

# model

if args.model =='SFIGF':
    model = SFIGF(ch=args.base_channel) 
    
else:
    raise NotImplementedError(f'Model {args.model} not found')

# loss
if args.loss == 'L1':
    criterion = nn.L1Loss()
elif args.loss == 'L2':
    criterion = nn.MSELoss()
else:
    raise NotImplementedError(f'Loss {args.loss} not found')

# dataset
if args.dataset == 'NYU':
    dataset = NYUDataset
elif args.dataset == 'Lu':
    dataset = LuDataset
elif args.dataset == 'Middlebury':
    dataset = MiddleburyDataset
elif args.dataset == 'NoisyMiddlebury':
    dataset = NoisyMiddleburyDataset
else:
    raise NotImplementedError(f'Dataset {args.loss} not found') 

if args.model in ['SFIGF']:
    if not args.test:
       
        train_dataset = dataset(root=args.data_root, split='train', scale=args.scale, downsample=args.interpolation, augment=True, pre_upsample=args.no_pre_upsample, input_size=args.input_size, noisy=args.noisy)
    test_dataset = dataset(root=args.data_root, split='test', scale=args.scale, downsample=args.interpolation, augment=False, pre_upsample=args.no_pre_upsample, noisy=args.noisy)

else:
    raise NotImplementedError(f'Dataset for model type {args.model} not found')

# dataloader
if not args.test:
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch, pin_memory=True, drop_last=False, shuffle=True, num_workers=args.num_workers)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch, pin_memory=True, drop_last=False, shuffle=False, num_workers=args.num_workers)

# trainer
if not args.test:
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    trainer = Trainer(args, args.name, model, objective=criterion, optimizer=optimizer, lr_scheduler=scheduler, metrics=[RMSEMeter(args)], device='cuda', use_checkpoint=args.checkpoint, eval_interval=args.eval_interval)
else:
    if args.model in ['GuidedFilter']:
        trainer = Trainer_GIF(args, args.name, model, objective=criterion, metrics=[RMSEMeter(args)], device='cuda', use_checkpoint=args.checkpoint)
    else:
        trainer = Trainer(args, args.name, model, objective=criterion, metrics=[RMSEMeter(args)], device='cuda', use_checkpoint=args.checkpoint)

# main
if not args.test:
    trainer.train(train_loader, test_loader, args.epoch)

if args.save:
    # save results (doesn't need GT)
    trainer.test(test_loader)
    trainer.evaluate(test_loader)
else:
    # evaluate (needs GT)
    trainer.evaluate(test_loader)
