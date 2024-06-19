from copy import deepcopy
import torch
import torch.nn as nn
from model.ResNet import ResNet
import torchvision.models as models
import numpy as np
import torch.nn.functional as F
import time
import os
from torchsummary import summary
from model.patch_var import Patch_Var
from model.prior import prior_query2_channel_last_L_attention_rm_SA

from model.Neck import RFB_Conv,RFB_Lite,RFB_ori,RFB_ori_Lite
from model.Decoder import my_aggregation_lite_CSP,SimpleDecoder
from model.EdgeRefine import Edge_Filter,EdgePro_Auxiliary,ReFine_CSP
def save_as_numpy(xs: list, name):
    for i, x in enumerate(xs):
        x = x.data.cpu().numpy()
        np.savez('./mid_feature/mid_result_%s_%d.npz' % (name, i), x)


class MEPNet(nn.Module):
    def __init__(self, cfg):
        super(MEPNet, self).__init__()
        RFB_m = eval(cfg.model.neck.type)
        self.save_mid = False

        self.resnet = ResNet()
        self.head1_rfb1 = RFB_m(
            256, cfg.model.mid_channel, using_self_attention=cfg.model.neck.using_PA,using_CA=cfg.model.neck.using_CA)
        self.head1_rfb2 = RFB_m(
            512, cfg.model.mid_channel, using_self_attention=cfg.model.neck.using_PA,using_CA=cfg.model.neck.using_CA)
        self.head1_rfb3 = RFB_m(
            1024, cfg.model.mid_channel, using_self_attention=cfg.model.neck.using_PA,using_CA=cfg.model.neck.using_CA)
        self.head1_rfb4 = RFB_m(
            2048, cfg.model.mid_channel, using_self_attention=cfg.model.neck.using_PA,using_CA=cfg.model.neck.using_CA)
        self.head1_agg1 = eval(cfg.model.header)(cfg)

        if cfg.model.Edge_Ass.using:
            if not cfg.model.Edge_Ass.using_canny:
                if  cfg.model.Edge_Ass.using_PV:
                    self.PV=Patch_Var(cfg.model.Edge_Ass.PV_patch_size, cfg.model.Edge_Ass.PV_reduction, cfg.model.Edge_Ass.PV_normal, cfg.model.Edge_Ass.gain)
                else:
                    self.conv = nn.Sequential(nn.Conv2d(cfg.model.Edge_Ass.inchannel,3,kernel_size=3,padding=1,bias=False),
                                        nn.BatchNorm2d(3),
                                        nn.Sigmoid())
            
            self.edge_aux = EdgePro_Auxiliary(cfg, 4, cfg.model.Edge_Ass.mid_channel)#eval(cfg.model.Edge_Ass.type)(4, cfg.model.Edge_Ass.mid_channel)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.cfg = cfg

        if self.training:
            self.initialize_weights()
        
    def info(self):
        print(self)

    def forward(self, x):  # 64x352x352
        # print(x.size())
        if self.cfg.model.Edge_Ass.using_canny:
            x,x_edge=x
            x=x_edge+x
        else:
            if self.cfg.model.Edge_Ass.using:
                x_edge=x
                if self.cfg.model.Edge_Ass.using_PV:
                #     # pv=self.PV(x)
                    x_edge=self.PV(x)
                #     # save_as_numpy([x_edge],'pv')
                #     # x_edge=torch.cat([x,pv],dim=1)        
                else:
                    x_edge = self.conv(x_edge)
                # x = x_edge+x
                if self.cfg.model.Edge_Ass.using_edge_SC:
                    # temp=self.conv(x_edge*x)
                    x = x_edge+x

        # save_as_numpy([x],'x')
        # save_as_numpy([temp],'temp')

        x0 = self.resnet.conv1(x)  # 64x156x156
        x0 = self.resnet.bn1(x0)  # 64x156x156
        x0 = self.resnet.relu(x0)  # 64x156x156

        x1 = self.resnet.maxpool(x0)  # 64x88x88
        x1 = self.resnet.layer1(x1)  # 256 x 88 x 88
        x2 = self.resnet.layer2(x1)  # 512 x 44 x 44
        x3 = self.resnet.layer3(x2)  # 1024 x 22 x 22
        x4 = self.resnet.layer4(x3)  # 2048 x 11 x 11
        x1 = self.head1_rfb1(x1)
        x2 = self.head1_rfb2(x2)
        x3 = self.head1_rfb3(x3)
        x4 = self.head1_rfb4(x4)

            
        result_dict = self.head1_agg1(x1, x2, x3, x4)
        

        attention_up = self.upsample(self.upsample(result_dict['merge_list'][-1]))

        if self.cfg.model.Edge_Ass.using:
            edge_x, attention_up = self.edge_aux(attention_up, x_edge)
            if not self.training:
                return attention_up[0]
            result_dict['merge_list'] = result_dict['merge_list']+attention_up
            result_dict['edge_result']=edge_x
        else:
            if not self.training:
                return attention_up
            result_dict['merge_list'][-1] = attention_up
        result_dict['merge_list']=result_dict['merge_list'][len(result_dict['merge_list'])-self.cfg.loss.loss_scale.number:]    
        return result_dict

    def initialize_weights(self):
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #         # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #         # nn.init.normal_(m.bias, std=1e-6)
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m,nn.Linear):
        #         nn.init.xavier_uniform_(m.weight)
        #         nn.init.normal_(m.bias, std=1e-6)
        res50 = models.resnet50(pretrained=True)
        pretrained_dict = res50.state_dict()
        # all_params = {}
        # for k, v in self.resnet.state_dict().items():
        #     if k in pretrained_dict.keys():
        #         v = pretrained_dict[k]
        #     elif 'conv1' in k:
        #         # name=k.split('_2')[0] + k.split('_2')[1]
        #         # a=pretrained_dict[name]
        #         # v=pretrained_dict[name][:3]
        #         pass
        #     elif 'bn1_2' in k:
        #         pass
        #         # name=k.split('_2')[0] + k.split('_2')[1]
        #         # if 'tracked' in k:
        #         #    v=pretrained_dict[name]
        #         # else:
        #         #    v=pretrained_dict[name][:3]
        #     elif '_1' in k:
        #         name = k.split('_1')[0] + k.split('_1')[1]
        #         v = pretrained_dict[name]
        #     elif '_2' in k:
        #         name = k.split('_2')[0] + k.split('_2')[1]
        #         v = pretrained_dict[name]
        #     all_params[k] = v
        # assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        del pretrained_dict['fc.weight']
        del pretrained_dict['fc.bias']
        self.resnet.load_state_dict(pretrained_dict)
        if self.cfg.model.PQ.using and self.cfg.model.PQ.using_pretrain:
            ckpt=torch.load(self.cfg.model.PQ.pretrain_path,map_location='cpu')['model']#
            PQ4_state_dict=self.head1_agg1.PQ4.state_dict()
            PQ_param={}
            for k,v in ckpt.items():
                if 'PQ' in k:
                    k=k.replace('PQ4.','')
                    if 'conv' in k:
                        # continue
                        v=PQ4_state_dict[k]
                    PQ_param[k]=v
            assert len(self.head1_agg1.PQ4.state_dict().keys())==len(PQ_param)
            if  self.cfg.model.PQ.position[0]:
                self.head1_agg1.PQ4.load_state_dict(PQ_param)
            if  self.cfg.model.PQ.position[1]:    
                self.head1_agg1.PQ3.load_state_dict(PQ_param)
            if  self.cfg.model.PQ.position[2]:    
                self.head1_agg1.PQ2.load_state_dict(PQ_param)
            # del res50
            del PQ4_state_dict
            del ckpt
            del PQ_param
            del pretrained_dict

def speed(model, name):
    t0 = time.time()
    data = np.random.uniform(-1, 1, [2, 3, 352, 352]).astype('float32')
    data = torch.Tensor(data).cuda()

    t1 = time.time()

    model(data)
    ts = []
    for i in range(100):
        t2 = time.time()

        model(data)
        t3 = time.time()
        ts.append(t3 - t2)

    print('%s : %fms' % (name, sum(ts) * 10))


# CPD_LOSS : 34
if __name__ == '__main__':
    os.chdir('../')
    model = MEPNet()
    model.cuda()
    print(summary(model, (3, 352, 352)))
    speed(model, 'resnet')

#图像相似度函数
