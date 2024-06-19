import os
from yacs.config import CfgNode as CN

_C = CN()
_C.gpus='0,1'
_C.local_rank=-1
_C.config_file=''
_C.global_rank=-1
_C.world_size=1
_C.sync_bn=True
_C.seed=1000
_C.print_rate=100
_C.empty_cache=False
_C.BalancedData=False
_C.model=CN()
_C.model.mid_channel=256
_C.model.expand=[1.0,1.0,1.0]
_C.model.using_split=False
_C.model.neck=CN()
_C.model.neck.type='RFB_Conv'
_C.model.neck.using_PA=False
_C.model.neck.using_CA=False
_C.model.Edge_Ass=CN()
_C.model.Edge_Ass.type='EdgePro_Auxiliary'
_C.model.Edge_Ass.using_probability=True
_C.model.Edge_Ass.using_canny=False
_C.model.Edge_Ass.using=False
_C.model.Edge_Ass.using_PV=False
_C.model.Edge_Ass.using_edge_SC=False
_C.model.Edge_Ass.inchannel=4 if _C.model.Edge_Ass.using_PV else 3
_C.model.Edge_Ass.PV_reduction='max'
_C.model.Edge_Ass.PV_patch_size=3
_C.model.Edge_Ass.PV_normal=False
_C.model.Edge_Ass.mid_channel=4
_C.model.Edge_Ass.gain=2.0


_C.model.PQ=CN()
_C.model.PQ.type='patch_query2_swin_channel_last.prior_query'
_C.model.PQ.using=False
_C.model.PQ.patch_size=2
_C.model.PQ.map_size=16
_C.model.PQ.num_head=4
_C.model.PQ.using_pretrain=False
_C.model.PQ.position=[True,False,False]
_C.model.PQ.pretrain_path='./models/pretrained/prior_query2_channel_last_L_attention_rm_SA/pretrain/model_199.pth'
_C.model.PQ.using_LA=True
_C.model.PQ.using_prior=True

_C.model.SR=CN() #salient object reconstruction
_C.model.SR.using=False
_C.model.header='my_aggregation_lite_CSP'

_C.model.EQ=CN()
_C.model.EQ.using=False
_C.model.EQ.map_size=88
_C.model.EQ.win_size=4
_C.model.EQ.num_heads=4

_C.model.AF=CN()
_C.model.AF.using=False
_C.model.AF.position=[True,False,False]


_C.dataloader=CN()
_C.dataloader.batch_size=16
_C.dataloader.num_work=8
_C.dataloader.train_size=352
_C.dataloader.test_size=352
_C.dataloader.using_random_size=True
_C.dataloader.train_path=['../SOD_Data/DUTS-TR/DUTS-TR-Image/', '../SOD_Data/DUTS-TR/DUTS-TR-Mask/']
_C.dataloader.test_path=['../SOD_Data/DUTS-TE/DUTS-TE-Image/', '../SOD_Data/DUTS-TE/DUTS-TE-Mask/']


_C.solver=CN()
_C.solver.type='AdamW'
_C.solver.epoch=70
_C.solver.momen=0.9
_C.solver.weight_decay=1e-5
_C.solver.lr=1e-4
_C.solver.base_batchsize=12
_C.solver.min_lr=_C.solver.lr*0.001
_C.solver.warmup_lr=_C.solver.lr*0.0001
_C.solver.init_epoch=10
_C.solver.warmup_epoch = 3
_C.solver.t_mul=2
_C.solver.using_clip=True
_C.solver.clip=0.5
_C.solver.lr_rate=[0.1,1,1]
_C.solver.decay_rate=0.5
_C.solver.lr_step=[10,30,70]


_C.loss=CN()
_C.loss.combination=['DICE','SSIM', 'WBCE']
_C.loss.combination_p=[[], [], []]
_C.loss.loss_weight=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
_C.loss.loss_scale=CN()
_C.loss.loss_scale.number=4
_C.loss.loss_scale.combination=[[1],[1,2],[1,2,3],[1,2,3]]
_C.loss.using_dice=True
_C.loss.edge=CN()
_C.loss.edge.using=_C.model.Edge_Ass.using
_C.loss.edge.gamma=1.5
_C.loss.edge.gains=[1,8,64]
_C.loss.edge.weights=[1,1/4,1/16]
_C.loss.edge.stage=[10,20,40]
_C.loss.edge.max_rate=0.9
_C.loss.edge.base_size=288
_C.loss.edge.loss_type='mse'
_C.loss.using_Deepest_loss=False
_C.loss.using_normal=True
_C.loss.using_filter_interpolate=False
_C.loss.mask_binary=False
_C.loss.AFloss=CN()
_C.loss.AFloss.weight=0.1
_C.loss.AFloss.Four=False
_C.loss.using_Rweight=True
_C.loss.Rweight=0.5



