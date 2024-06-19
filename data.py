import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2 as cv
import numpy as np
import torch
from utils import torch_distributed_zero_first
from tqdm import tqdm
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask):
        image = (image - self.mean) / self.std
        mask = mask / 255.0
        return image, mask


class RandomCrop(object):
    def __call__(self, image, mask):
        H, W, _ = image.shape
        randw = np.random.randint(W / 8)
        randh = np.random.randint(H / 8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H + offseth - randh, offsetw, W + offsetw - randw
        # crop_mask=mask[p0:p1, p2:p3]
        # if np.max(crop_mask)==0:
        #     print('get')
        #     return image, mask
        return image[p0:p1, p2:p3, :], mask[p0:p1, p2:p3]


class RandomFlip(object):
    def __call__(self, image, mask):
        if np.random.randint(2) == 0:
            return image[:, ::-1, :], mask[:, ::-1]
        else:
            return image, mask


class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask):
        image = cv.resize(image, dsize=(self.W, self.H), interpolation=cv.INTER_LINEAR)
        mask = cv.resize(mask, dsize=(self.W, self.H), interpolation=cv.INTER_LINEAR)
        return image, mask


class ToTensor(object):
    def __call__(self, image, mask):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        mask = torch.from_numpy(mask)
        return image, mask


class SalObjDataset(data.Dataset):
    def __init__(self, cfg,image_root, gt_root, trainsize, mode='train'):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.images = sorted(self.images)  # [:len(self.images)//10]
        self.gts = sorted(self.gts)  # [:len(self.gts)//10]
        self.filter_files()

        self.normalize = Normalize(mean=np.array([[[124.55, 118.90, 102.94]]]), std=np.array([[[56.77, 55.97, 57.50]]]))
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()
        self.resize = Resize(trainsize, trainsize)
        self.totensor = ToTensor()
        self.using_random_size=cfg.dataloader.using_random_size
        self.cfg=cfg
        self.mode = mode

    def __getitem__(self, index):
        ori_image = cv.imread(self.images[index])
        image=ori_image[:, :, ::-1].astype(np.float32)
        mask = cv.imread(self.gts[index], 0).astype(np.float32)
        shape = mask.shape
        if self.mode == 'test':
            image, mask = self.normalize(image, mask)
            image, mask = self.resize(image, mask)
            image, mask = self.totensor(image, mask)
            if self.cfg.model.Edge_Ass.using_canny:
                ori_image=cv.resize(ori_image,(self.trainsize,self.trainsize), interpolation=cv.INTER_LINEAR)
                edge=[]
                for j in range(3):
                    edge.append(cv.Canny(np.array(ori_image[:,:,j],dtype=np.uint8),50,200))
                    edge[j]=np.array(edge[j],dtype=np.float)[None,:,:]/255
                edge=torch.from_numpy(np.concatenate(edge,axis=0))
                edge=np.array(edge,dtype=np.float)/255
                edge=torch.from_numpy(edge)
                return image.float(),edge.float(), mask.float(), self.gts[index].split('/')[-1], shape
            return image.float(), mask.float(), self.gts[index].split('/')[-1], shape
        elif self.mode == 'val':
            image, mask = self.normalize(image, mask)
            image, mask = self.resize(image, mask)
            image, mask = self.totensor(image, mask)
            if self.cfg.model.Edge_Ass.using_canny:
                ori_image=cv.resize(ori_image,(self.trainsize,self.trainsize), interpolation=cv.INTER_LINEAR)
                edge=[]
                for j in range(3):
                    edge.append(cv.Canny(np.array(ori_image[:,:,j],dtype=np.uint8),50,200))
                    edge[j]=np.array(edge[j],dtype=np.float)[None,:,:]/255
                edge=torch.from_numpy(np.concatenate(edge,axis=0))
                edge=np.array(edge,dtype=np.float)/255
                edge=torch.from_numpy(edge)
                return image.float(),edge.float(), mask.float()
            return image.float(), mask.float()
        else:
            # image, mask = self.normalize(image, mask)
            image, mask = self.randomcrop(image, mask)
            image, mask = self.randomflip(image, mask)
            return image, mask

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def collate(self, batch):
        if self.using_random_size:
            size1 = [192, 224, 256, 288, 320, 352, 384, 416][np.random.randint(8)]
            size2 = [192, 224, 256, 288, 320, 352, 384, 416][np.random.randint(8)]
        else:
            size1=352
            size2=352
        images, masks = [list(item) for item in zip(*batch)]
        edges=[]
        S_imgs=[]
        Iedges=[]
        for i in range(len(batch)):
            images[i] = cv.resize(images[i], dsize=(size1, size2), interpolation=cv.INTER_LINEAR)
            masks[i] = cv.resize(masks[i], dsize=(size1, size2), interpolation=cv.INTER_LINEAR)
            
            if self.cfg.model.SR.using:
                mask=cv.resize(masks[i], dsize=(size1//4, size2//4), interpolation=cv.INTER_LINEAR)/255.0
                S_img=cv.resize(images[i], dsize=(size1//4, size2//4), interpolation=cv.INTER_LINEAR)
                S_img[mask<=0,:]=0
                S_imgs.append(S_img/255.0)
            
            if self.cfg.model.Edge_Ass.using_canny:
                Cedge=[]
                for j in range(3):
                    Cedge.append(cv.Canny(np.array(images[i][:,:,j],dtype=np.uint8),50,200))
                    Cedge[j]=np.array(Cedge[j],dtype=np.float)[None,:,:]/255
                Cedge=torch.from_numpy(np.concatenate(Cedge,axis=0))
                Iedges.append(Cedge)
            
            images[i],masks[i]=self.normalize(images[i],masks[i])
            
            if self.cfg.model.Edge_Ass.using:
                mask=np.copy(masks[i])
                mask[mask > 0.5] = 1
                mask[mask <= 0.5] = 0
                mask=(mask*255).astype(np.uint8)
                contours,_=cv.findContours(mask,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)
                edge=np.zeros_like(mask,dtype=np.uint8)
                edge=cv.drawContours(edge,contours,-1,255,1)
                if self.cfg.model.Edge_Ass.using_probability:
                    mask=cv.copyMakeBorder(mask, 3, 3, 3, 3, cv.BORDER_CONSTANT, value=0)
                    edge=edge*1.0+np.abs(mask*1.0-cv.blur(mask,ksize=(7,7))*1.0)[3:-3, 3:-3]
                edges.append(edge/np.max(edge))
                
        output=[]
        images = torch.from_numpy(np.stack(images, axis=0)).permute(0, 3, 1, 2).float()
        output.append(images)
        
        masks = torch.from_numpy(np.stack(masks, axis=0)).unsqueeze(1).float()
        output.append(masks)
        
        if self.cfg.model.SR.using:
            S_imgs = torch.from_numpy(np.stack(S_imgs, axis=0)).permute(0, 3, 1, 2).float()
        output.append(S_imgs)
            
        if self.cfg.model.Edge_Ass.using:
            edges=torch.from_numpy(np.stack(edges, axis=0)).unsqueeze(1).float()
        output.append(edges)
        
        if self.cfg.model.Edge_Ass.using_canny:
            Iedges=torch.from_numpy(np.stack(Iedges, axis=0)).float()
        output.append(Iedges)
        
        return output

    def __len__(self):
        return len(self.images)


# lmdb+mixup_fn 31,33,30

def get_loader(cfg,image_root, gt_root, batchsize, trainsize, local_rank, mode='train',  num_workers=1, pin_memory=True):
    with torch_distributed_zero_first(local_rank):
        dataset = SalObjDataset(cfg,image_root, gt_root, trainsize, mode=mode)
        
    if mode == 'train':
        sampler = torch.utils.data.distributed.DistributedSampler(dataset,seed=cfg.seed,drop_last=True) if local_rank != -1 else None
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=batchsize,
                                      shuffle=False,
                                      num_workers=num_workers,
                                      sampler=sampler,
                                      pin_memory=pin_memory, collate_fn=dataset.collate)
    else:
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=batchsize,
                                      shuffle=False,
                                      num_workers=num_workers,
                                      pin_memory=pin_memory)
    return data_loader


class picture_process(object):

    def __init__(self, testsize):
        self.testsize = testsize
        self.normalize = Normalize(mean=np.array([[[124.55, 118.90, 102.94]]]), std=np.array([[[56.77, 55.97, 57.50]]]))
        self.resize = Resize(testsize, testsize)
        self.totensor = ToTensor()

    def get_data(self, image):
        image, mask = self.normalize(image, image[:, :, 0])
        image, mask = self.resize(image, mask)
        image, mask = self.totensor(image, mask)
        return image.float()

if __name__=="__main__":
    dataloader=get_loader("../SOD_Data/DUTS-TR/DUTS-TR-Image/", '../SOD_Data/DUTS-TR/DUTS-TR-Mask/', 8, 352, -1, mode='test',  num_workers=8, pin_memory=True)
    for i, pack in enumerate(tqdm(dataloader)):
    # images, gts = pack
        images, gts, edge_x = pack