import torch
import torch.nn.functional as F
from torch.autograd import Variable
from model.MEPNet import MEPNet
from data import get_loader
import matplotlib.pyplot as plt
import os
import collections
import numpy as np
import cv2 as cv
import pickle
from tqdm import tqdm
import time
import shutil
keys = 'RES_Model'


def draw_feature():
    folder='./mid_feature'
    save_path='./mid_feature_2024_03_13_01_33_08'
    files=os.listdir(folder)
    for file in tqdm(files):
        tag=file.split('result_')[-1].split('.')[0]
        if os.path.exists(os.path.join(save_path,tag)):
            shutil.rmtree(os.path.join(save_path,tag))
        os.makedirs(os.path.join(save_path,tag))
        data=np.load(os.path.join(folder,file))['arr_0']
        for i in range(len(data)):
            part_data=data[i]
            if len(part_data.shape)==2:
                part_data=part_data[None,:,:]
            if part_data.shape[2]==3:
                img=part_data
                
                img=np.array(img*255,dtype=np.uint8)
                plt.imshow(img)
                plt.savefig(os.path.join(save_path,tag,('%d_%d.png'%(i,0)).zfill(7)))
                cv.imwrite(os.path.join(save_path,tag,('%d_%d.jpg'%(i,0)).zfill(7)),img)
            else:
                for j in range(len(part_data)):
                    img=part_data[j,:,:]
                    img=np.array((img-np.min(img))/(np.max(img)-np.min(img))*255,dtype=np.uint8)
                    # img=np.array(img*255,dtype=np.uint8)
                    cv.imwrite(os.path.join(save_path,tag,('%d_%d.jpg'%(i,j)).zfill(7)),img)
                # img=part_data[j,:,:]
                #plt.imshow(img)
                #plt.savefig(os.path.join(save_path,tag,('%d_%d.jpg'%(i,j)).zfill(7)))
                #plt.close()
                # img=np.array((img-np.min(img))/(np.max(img)-np.min(img))*255,dtype=np.uint8)
                
            #plt.imshow(img)
            #plt.savefig(os.path.join(save_path,tag,('%d.jpg'%i).zfill(7)))
            #plt.close()

if __name__ == '__main__':
    draw_feature()
