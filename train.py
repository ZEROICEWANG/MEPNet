from operator import le
import os
import random
import sys
import time
from datetime import datetime
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

from data import get_loader
from loss.loss import LOSS
from model.MEPNet import MEPNet
from save_log import *
from utils import  clip_gradient, save_py, seed_torch,update_lr,select_device,set_lr
from timm.scheduler.cosine_lr import CosineLRScheduler
import argparse
from config import cfg
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from predict import process_dict
from torch.utils.tensorboard import SummaryWriter

def validate(data_loader,model, keys, time_, epoch,save_point):
    '''model.eval()
    acces1 = []
    acces2 = []
    steps = len(data_loader)
    with torch.no_grad():
        for i, pack in enumerate(data_loader):
            images, gts = pack
            images = Variable(images)
            gts = Variable(gts)
            images = images.cuda()
            gts = gts.cuda()
            # optimizer.zero_grad()
            atts, dets = model(images)
            acc1 = 0
            acc2 = 0
            gt = gts.data.cpu().numpy().squeeze()

            at = atts.sigmoid().data.cpu().numpy().squeeze()
            at = (at - at.min()) / (at.max() - at.min() + 1e-20)

            dt = dets.sigmoid().data.cpu().numpy().squeeze()
            dt = (dt - dt.min()) / (dt.max() - dt.min() + 1e-20)
            acc1 += 1 - np.mean(np.abs(at - gt))
            acc2 += 1 - np.mean(np.abs(dt - gt))
            acces1.append(acc1)
            acces2.append(acc2)
            if (i + 1) % 100 == 0 or (i + 1) == len(data_loader):
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], acc1: {:.4f}, acc2: {:.4f}'.
                      format(datetime.now(), epoch, opt.epoch, i + 1, steps, np.mean(acces1), np.mean(acces2)))'''

    acc = 1  # np.mean(acces2)
    '''if acc > best_acc:
        best_acc = acc
        state = {
            'model': model.state_dict(),
            'best_acc': best_acc,
            'epoch': epoch,
            'opt': None#optimizer.state_dict()
        }
        torch.save(state, os.path.join('./models', keys, time_, 'model_best_acc.pth'))'''
    if epoch >= save_point:
        model.eval()
        state = {
            'model': model.state_dict(),
            'best_acc': 1,
            'epoch': 23,
            'opt': None  # optimizer.state_dict()
        }
        torch.save(state, os.path.join(
            './models', keys, time_, 'model_%d.pth' % epoch))
    return best_acc, acc

def add_logs(writer:SummaryWriter,type,log_items,log_values,index):
    assert len(log_items)==len(log_values),'log items dont match log values, log items:%d, log values:%d'%(len(log_items,log_values))
    log_dict={log_items[i]:log_values[i] for i in range(len(log_items))}

    writer.add_scalars(type,log_dict,index)

def train(train_loader, model, optimizer, epoch, loss_f, lr_scheduler, cfg,device,writer):
    if cfg.local_rank in [0,-1]:

        loss_item = [''.join(['L%d-%s:{:.4f}, ' % (i,cfg.loss.combination[j-1][0]) for j in cfg.loss.loss_scale.combination[i-1]]) for i in range(1, cfg.loss.loss_scale.number+1)]
        if cfg.model.SR.using:
            loss_item = loss_item+['L4-mse:{:.4f}, ']
        if cfg.loss.edge.using:
            loss_item = loss_item+['L4-e:{:.4f}, ','L4-eI:{:.4f}, ']
        if cfg.model.AF.using:
            index=np.argwhere(np.array(cfg.model.AF.position))
            base=['AF1:{:.4f}, ','AF2:{:.4f}, ','AF3:{:.4f}, ']
            loss_item = loss_item+[base[i[0]] for i in index]
        msg_item = '{} Ep [{:03d}/{:03d}], Step [{:04d}/{:04d}], Lr1:{:.4e}, Lr2:{:.4e}, Lr3:{:.4e}, ' + \
            ''.join(loss_item)+'Los:{:.4f}'
        loss_item=[item.strip().split(',') for item in loss_item]
        loss_item = [item.strip().split(':{')[0] for sublist in loss_item for item in sublist if len(item)!=0]+['Los']
        lrs_item=['Lr1', 'Lr2', 'Lr3']
    model.train()
    losses = []
    losses_list = []
    num_iter = len(train_loader)
    for i, pack in enumerate(train_loader):
        images = Variable(pack[0]).contiguous().to(device)
        gt_dict={}
        gts = Variable(pack[1]).contiguous().to(device)
        gt_dict['gts']=gts
        if cfg.model.SR.using:
            Salient_image=Variable(pack[2]).contiguous().to(device)
            gt_dict['SR']=Salient_image
        if cfg.model.Edge_Ass.using:
            edge_x = Variable(pack[3]).contiguous().to(device)
            gt_dict['edge']=edge_x
        if cfg.model.Edge_Ass.using_canny:
            Input_edge = Variable(pack[4]).contiguous().to(device)
            images=[images,Input_edge]

        optimizer.zero_grad()

        results_dict = model(images)

        loss, loss_list = loss_f(
                results_dict, gt_dict)

        loss.backward()
        if cfg.solver.using_clip:
            clip_gradient(optimizer, cfg.solver.clip)
        optimizer.step()
        lr_scheduler.step_update(epoch*num_iter+i)
        losses.append(loss.item())
        losses_list.append(loss_list)
        if cfg.local_rank in [0,-1]:
            if i==0 or (i+1) % cfg.print_rate == 0 or (i + 1) == total_step:
                lrs = [optimizer.param_groups[i]['lr'] for i in range(3)]
                loss_value=np.mean(losses_list, axis=0).tolist()+[np.mean(losses)]
                print(msg_item.format(str(datetime.now()).split('.')[
                    0], epoch, cfg.solver.epoch, i + 1, total_step, *lrs, *loss_value))
                if i==0:
                    continue
                add_logs(writer,'lrs',lrs_item,lrs,epoch*total_step+i)
                add_logs(writer,'loss',loss_item,loss_value,epoch*total_step+i)
            if (i + 1) == total_step:
                add_logs(writer,'loss_epo',loss_item,loss_value,epoch*total_step+i)
                add_logs(writer,'epoch_iter',['epoch'],[epoch],epoch*total_step+i)
        if cfg.empty_cache:
            torch.cuda.empty_cache()
    return sum(losses) / len(losses)


iter = 0
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PyTorch Object Detection Training")
    parser.add_argument("--config-file",default="./config/standard.yaml",metavar="FILE",help="path to config file",type=str,)
    parser.add_argument("--gpus",default="0",metavar="FILE",help="path to config file",type=str,)
    parser.add_argument("--local_rank",default=-1,metavar="FILE",help="path to config file",type=int,)
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.local_rank=args.local_rank
    cfg.sync_bn=args.sync_bn
    cfg.gpus=args.gpus
    update_lr(cfg)
    device = select_device(cfg.gpus, batch_size=cfg.dataloader.batch_size)
    seed_torch(cfg.seed)
    
    if cfg.local_rank != -1:
        assert torch.cuda.device_count() > cfg.local_rank
        torch.cuda.set_device(cfg.local_rank)
        device = torch.device('cuda', cfg.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        cfg.world_size = dist.get_world_size()
        cfg.global_rank = dist.get_rank()
        assert cfg.dataloader.batch_size % cfg.world_size == 0, '--batch-size must be multiple of CUDA device count'
        cfg.dataloader.batch_size = cfg.dataloader.batch_size // cfg.world_size
    
    cfg.freeze()

    keys = 'RES_Model'
    best_acc = 1
    time_ = str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    
    model_name = '1111'
    writer=None
    if cfg.global_rank in [-1, 0]:
        if not os.path.exists(os.path.join('./models', keys, time_)):
            os.makedirs(os.path.join('./models', keys, time_))
        if not os.path.exists(os.path.join('./logs',  time_)):
            os.makedirs(os.path.join('./logs',  time_))
        if not os.path.exists(os.path.join('./logs',  time_, 'files')):
            os.makedirs(os.path.join('./logs',  time_, 'files'))
        sys.stdout = Logger(os.path.join(
        './logs', time_, 'train_' + keys + time_ + '.txt'))
        file=os.path.join(
            './logs', time_, 'env.txt')
        os.system('conda info -e >> %s'%file)
        os.system('pip list >> %s'%file)
        
        print('saving files')
        save_py(os.path.join('./logs',  time_, 'files'), './')
        print('saving files finished')
        save_point=cfg.solver.epoch-10
        if iter > 0:
            print(
                'describe:full test data, full train data, train test add conv in refine,restart %s' % model_name)
            save_point=cfg.solver.epoch-10
        else:
            print(args.config_file,'\n','ablation study')
        writer=SummaryWriter(log_dir=os.path.join('./logs', time_, 'train_' + keys + time_ + '.logs'),flush_secs=5)

    best_acc = 0
    best_epoch = 0


    loss_f = LOSS(cfg)
    best_loss = 1e26

    
    model = MEPNet(cfg)
    
    if iter > 0:
        dic = torch.load(os.path.join('./models', keys,
                         model_name, 'model_69.pth'),map_location='cpu')
        model.load_state_dict(process_dict((dic['model'])))
        best_loss = 0  # dic["best_loss"]
        best_acc = 0  # dic["best_acc"]d
        best_epoch = dic['epoch']
        base_bc, de_p,tr_p = [], [], []
        for name, param in model.named_parameters():
            if 'resnet' in name:
                base_bc.append(param)
            elif 'PQ' in name:
                tr_p.append(param)
            else:
                de_p.append(param)
        optimizer = torch.optim.AdamW([{'params': base_bc}, {'params': tr_p}, {'params': de_p}],
                                      cfg.solver.lr, weight_decay=cfg.solver.weight_decay)

        print(
            'best epoch:%d, best loss:%.4f,best acc:%.4f' % (best_epoch, best_loss, best_acc))
        del dic
    else:
        base_bc, de_p,tr_p = [], [], []
        for name, param in model.named_parameters():
            if 'resnet' in name:
                base_bc.append(param)
            elif 'PQ' in name:
                tr_p.append(param)
            else:
                de_p.append(param)
        optimizer = torch.optim.AdamW([{'params': base_bc}, {'params': tr_p}, {'params': de_p}],
                                      cfg.solver.lr, weight_decay=cfg.solver.weight_decay)
    model.to(device)
    set_lr(optimizer, cfg.solver.lr,cfg)
    cuda = device.type != 'cpu'
    if cuda and cfg.global_rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    if cfg.sync_bn and cuda and cfg.global_rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        print('Using SyncBatchNorm()')
    
    if cuda and cfg.global_rank != -1:
        model = DDP(model, device_ids=[cfg.local_rank], output_device=(cfg.local_rank),find_unused_parameters=True)
    
    
    train_loader = get_loader(cfg,cfg.dataloader.train_path[0], cfg.dataloader.train_path[1],
                              batchsize=cfg.dataloader.batch_size,
                              trainsize=cfg.dataloader.train_size, local_rank=cfg.local_rank,mode='train', num_workers=cfg.dataloader.num_work)
    test_loader = get_loader(cfg,cfg.dataloader.train_path[0], cfg.dataloader.train_path[1],
                             batchsize=cfg.dataloader.batch_size,
                             trainsize=cfg.dataloader.test_size,local_rank=cfg.local_rank, mode='val')
    total_step = len(train_loader)
    
    lr_scheduler = CosineLRScheduler(optimizer, t_initial=cfg.solver.init_epoch*len(train_loader), t_mul=cfg.solver.t_mul, lr_min=cfg.solver.min_lr,
                                     warmup_lr_init=cfg.solver.warmup_lr, warmup_t=cfg.solver.warmup_epoch*len(train_loader), t_in_epochs=False,decay_rate=cfg.solver.decay_rate)
    if cfg.global_rank in [-1, 0]:
        print(cfg)
        print("Start training!")

    # print('update edge loss gain as ', loss_f.edge_loss.gain)
    
    lr_step=cfg.solver.lr_step
    for epoch in range(cfg.solver.epoch):
        if cfg.global_rank != -1:
            train_loader.sampler.set_epoch(epoch)
        if cfg.loss.edge.using:
            loss_f.edge_loss.step()
        loss = train(train_loader, model, optimizer,
                     epoch, loss_f, lr_scheduler, cfg,device,writer)
        if cfg.local_rank in [0,-1]:
            best_acc, acc = validate(test_loader,model, keys, time_, epoch,save_point=save_point)
            if loss < best_loss:
                best_loss = loss
            if best_acc == acc:
                best_epoch = epoch
            print('{} Epoch [{:03d}/{:03d}], Loss1: {:.4f}  acc: {:0.4f} best_acc: {:0.4f} best_epoch: {:}'.
              format(datetime.now(), epoch, cfg.solver.epoch, loss, acc, best_acc, best_epoch))
    if cfg.local_rank in [0,-1]:
        writer.close()
