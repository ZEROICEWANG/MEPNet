import collections
import os
import pickle
from re import L
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
import seaborn as sns
from data import get_loader, picture_process
from model.MEPNet import MEPNet
from config import cfg


def process_dict(stage_dict: dict):
    new_stage_dict = collections.OrderedDict()
    for keys in stage_dict.keys():
        new_keys = keys.split('module.')[-1]
        new_stage_dict[new_keys] = stage_dict[keys]
    return new_stage_dict


def save_with_plt(map,save_path):
    plt.imshow(map,cmap='rainbow')
    plt.savefig(save_path)

def save_with_cv(map,save_path):
    cv.imwrite(save_path, map)

def save_with_seaborn(data,save_path):
    sns.heatmap(data=data, vmax=np.max(data), vmin=np.min(data), cmap=plt.cm.jet, cbar=False)
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(save_path)
    plt.close()

def predict_img(Model_class,cfg,args):

    img=cv.imread(args.image_path)
    PP=picture_process(args.image_size,cfg)
    model = Model_class(cfg,False,False)
    model.cuda()
    dic = torch.load(os.path.join(args.model_weight))
    model.load_state_dict(process_dict(dic['model']))
    model.eval()
    
    with torch.no_grad():
        pack=PP.get_data(img)
        images=pack[0]
        images = Variable(images)
        images = images.cuda()
        dts_dict = model(images)
        for k,dts in dts_dict.items():
            if not isinstance(dts,list):
                dts=[dts]
            for i,dt in enumerate(dts):
                dt = dt.sigmoid().data.cpu().numpy().squeeze()

                map = np.around(dt*255)
                map=cv.resize(map,(352,352))
                save_path=str(args.image_path).replace('.j','_%s_%d.j'%(k,i))
                save_with_seaborn(map,save_path)
        torch.save(dts_dict,str(args.image_path).replace('.jpg','.pth'))
                



os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if __name__ == '__main__':
    parser = argparse.ArgumentParser('predict script', add_help=False)
    parser.add_argument('--model_weight', type=str,
                        default=r'models\RES_Model\standard\model.pth',
                        help='path to dataset folder', )
    parser.add_argument('--model',default='MEPNet')
    parser.add_argument(
        "--image_path",
        default="./data/ILSVRC2012_test_00000003.jpg",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--config-file",
        default="./config\standard.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--image_size",
        default=352,
        metavar="FILE",
        help="path to config file",
        type=float,
    )
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    
    predict_img(MEPNet,cfg,args)

    
