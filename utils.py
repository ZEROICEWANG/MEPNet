import os
import shutil
import torch
import numpy as np
import random
from contextlib import contextmanager
def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def set_lr(optimizer, init_lr,cfg):
    rate = cfg.solver.lr_rate
    print(rate)
    for i in range(len(rate)):
        optimizer.param_groups[i]['lr'] = init_lr *  rate[i]
        
            
    

def showing_lr(optimizer):
    for i in range(len(optimizer.param_groups)):
        print('update learning rates if group {:3d} to {:.4e}'.format(i, optimizer.param_groups[i]['lr']))

def save_py(save_dir,root,ignoring_dirs=['__pycache__','.idea','logs','logs_history','predict_result','predict_result2','predict_result_r','models']):
    files=os.listdir(root)
    for file in files:
        if file.endswith('.py') or file.endswith('.yaml') or file.endswith('.sh'):
            shutil.copyfile(os.path.join(root,file),os.path.join(save_dir,file))
        elif os.path.isdir(os.path.join(root,file)):
            if file in ignoring_dirs:
                continue
            save_py(save_dir,os.path.join(root,file))

def update_lr(cfg):
    cfg.solver.lr *= cfg.dataloader.batch_size/cfg.solver.base_batchsize
    cfg.solver.min_lr=cfg.solver.lr*0.001
    cfg.solver.warmup_lr=cfg.solver.lr*0.0001


def seed_torch(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) #设置单GPU种子
    torch.cuda.manual_seed_all(seed) #设置多GPU种子
    # torch.backends.cudnn.deterministic = True #返回默认卷积实现方式
    # torch.backends.cudnn.benchmark = False #禁止根据网络结构对卷积实现方式进行优化
    # torch.backends.cudnn.enabled = False #禁止使用非确定性算法
 
# seed_torch(args_dic['seed'])

#coding=utf-8
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def split_map(datapath):
    print(datapath)
    for name in tqdm(os.listdir(datapath+'/DUTS-TR-Mask')):
        mask = cv2.imread(datapath+'/DUTS-TR-Mask/'+name,0)
        body = cv2.blur(mask, ksize=(5,5))
        body = cv2.distanceTransform(body, distanceType=cv2.DIST_L2, maskSize=5)
        body = body**0.5

        tmp  = body[np.where(body>0)]
        if len(tmp)!=0:
            body[np.where(body>0)] = np.floor(tmp/np.max(tmp)*255)

        if not os.path.exists(datapath+'/body-origin/'):
            os.makedirs(datapath+'/body-origin/')
        cv2.imwrite(datapath+'/body-origin/'+name, body)

        if not os.path.exists(datapath+'/detail-origin/'):
            os.makedirs(datapath+'/detail-origin/')
        cv2.imwrite(datapath+'/detail-origin/'+name, mask-body)


def test():
    img=cv2.imread('./data/2.png',0)
    img_b = cv2.blur(img, ksize=(5,5))
    img_d = cv2.distanceTransform(img_b, distanceType=cv2.DIST_L2, maskSize=5)
    img_d5 = img_d**0.5

    tmp  = img_d5[np.where(img_d5>0)]
    body=np.copy(img_d5)
    if len(tmp)!=0:
        body[np.where(body>0)] = np.floor(tmp/np.max(tmp)*255)
    plt.figure(0)
    plt.imshow(img_b)
    plt.savefig('./1.png')
    plt.figure(0)
    plt.imshow(img_d)
    plt.savefig('./2.png')
    plt.figure(0)
    plt.imshow(img_d5)
    plt.savefig('./3.png')
    plt.figure(0)
    plt.imshow(tmp)
    plt.savefig('./4.png')
    plt.figure(0)
    plt.imshow(body)
    plt.savefig('./5.png')
    
def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = 'Using CUDA '
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            print("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                  (s, i, x[i].name, x[i].total_memory / c))
    else:
        print('Using CPU')

    print('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')    
    
@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()

if __name__=='__main__':
    # split_map('../SOD_Data/DUTS-TR')
    test()

