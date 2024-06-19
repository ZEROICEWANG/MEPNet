import collections
import os
import pickle
import time
from multiprocessing import Process, Queue
import argparse
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

from data import get_loader
from model.MEPNet import MEPNet
from config import cfg
keys = 'RES_Model'
import torch.multiprocessing


def process_dict(stage_dict: dict):
    new_stage_dict = collections.OrderedDict()
    for keys in stage_dict.keys():
        new_keys = keys.split('module.')[-1]
        new_stage_dict[new_keys] = stage_dict[keys]
    return new_stage_dict


def predict_img(queue,name,Model_class,cfg):
    files = os.listdir('./models/' + keys + '/' + name)
    files=sorted(files)
    for file in files:
        # index = j
        #name = '2023_03_26_11_13_20'
        file_dict = {1: 'DUT-OMRON', 2: 'HKU-IS',
                    3: 'PASCAL-S', 4: 'ECSSD', 5: 'DUTS'}
        model = Model_class(cfg)
        model.cuda()
        
        dic = torch.load(os.path.join('./models', keys, name, file))
        model.load_state_dict(process_dict(dic['model']))
        model.eval()
        batch_size = 16
        size=352
        save_file = name + '_' + file.split('.')[0].split('_')[-1]
        print(save_file)
        if not os.path.exists(os.path.join('./predict_result', save_file)):
            os.mkdir(os.path.join('./predict_result', save_file))
        counter = 0
        for key in range(1, 2):
            if key == 1:
                test_loader = get_loader(cfg,'../SOD_Data/DUT-OMRON/', '../SOD_Data/DUT-OMRON-GT/', batchsize=batch_size,
                                         trainsize=size, local_rank=-1, mode='test',num_workers=8)
                
            elif key == 2:
                test_loader = get_loader(cfg,'../SOD_Data/HKU-IS/', '../SOD_Data/HKU-IS-GT/', batchsize=batch_size,
                                         trainsize=size, local_rank=-1, mode='test',num_workers=8)
            elif key == 3:
                test_loader = get_loader(cfg,'../SOD_Data/PASCAL-S/', '../SOD_Data/PASCAL-S-GT/', batchsize=batch_size,
                                         trainsize=size, local_rank=-1, mode='test',num_workers=8)
            elif key == 4:
                test_loader = get_loader(cfg,'../SOD_Data/ECSSD/', '../SOD_Data/ECSSD-GT/', batchsize=batch_size,
                                         trainsize=size, local_rank=-1, mode='test',num_workers=8)
            elif key == 5:
                test_loader = get_loader(cfg,'../SOD_Data/DUTS-TE/DUTS-TE-Image/', '../SOD_Data/DUTS-TE/DUTS-TE-Mask/',
                                         batchsize=batch_size,
                                         trainsize=size, local_rank=-1, mode='test',num_workers=8)
                
            else:
                print('key error')
                return None

            if not os.path.exists(os.path.join('predict_result', save_file, file_dict[key])):
                os.mkdir(os.path.join('predict_result',
                         save_file, file_dict[key]))

            with torch.no_grad():
                bar=tqdm(test_loader)
                for i, pack in enumerate(bar):
                    images=pack[0]
                    images = Variable(images)
                    images = images.cuda()
                    if cfg.model.Edge_Ass.using_canny:
                        Input_edge,_,names,shape=pack[1:]
                        Input_edge = Variable(Input_edge)
                        Input_edge = Input_edge.cuda()
                        images=[images,Input_edge]
                    else:
                        _, names, shape = pack[1:]
                                       
                    # atts, dets = model(images)
                    dt = model(images)
                    
                    
                    queue.put([save_file, key, names, dt])
                    '''for i in range(len(dt)):
                        counter += 1
                        map = np.around(dt[i] * 255)
                        cv.imwrite(os.path.join('predict_result', save_file, file_dict[key],
                                                names[i].split('.')[0] + '.png'), map)'''
                    bar.set_description(file_dict[key])
                    #time.sleep(10)
            del test_loader
            # print(counter, ' ', np.mean(np.abs(gts[i].numpy().squeeze() - dt[i])), ' ', names[i])
    queue.put(['', '', '', ''])


def save_imgs(queue):
    file_dict = {1: 'DUT-OMRON', 2: 'HKU-IS',
                 3: 'PASCAL-S', 4: 'ECSSD', 5: 'DUTS'}
    while True:
        data = queue.get(block=True)
        save_file, key, names, dt = data
        if isinstance(dt,str):
            break
        dt = dt.sigmoid().data.cpu().numpy()[:,0]#.squeeze()
        for i in range(len(dt)):
            map = np.around(dt[i]*255)
            cv.imwrite(os.path.join('predict_result', save_file, file_dict[key],
                                    names[i].split('.')[0] + '.png'), map)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if __name__ == '__main__':
    parser = argparse.ArgumentParser('predict script', add_help=False)
    parser.add_argument('--name', type=str,
                        default='standard',
                        help='path to dataset folder', )
    parser.add_argument('--model',default='CPD_RES_PA')
    parser.add_argument(
        "--config-file",
        default="./config/standard.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    args = parser.parse_args()
    torch.multiprocessing.set_sharing_strategy('file_system')
    cfg.merge_from_file(args.config_file)
    
    args = parser.parse_args()
    start = time.time()
    name = args.name
    queue = Queue()
    p1 = Process(target=predict_img, args=(queue,name,MEPNet,cfg))
    p2 = Process(target=save_imgs, args=(queue,))
    p1.start()
    p2.start()
    p2.join()
    print(time.time() - start)