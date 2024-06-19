# [MEPNet: Mask Prior-distribution Interaction and Edge Probability Estimation for Salient Object Detection]

by XX

## Introduction
Salient Object Detection (SOD) has been researched extensively and achieved impressive performance. However, current methods still present defective results while facing complex scenes. Because the deceptive background hinders the methods from distinguishing the salient subject and presenting discriminative edges. To address this issue, we propose a method named MEPNet to realize two tasks: interacting the mask prior-distribution with multi-scale feature to suppress the background disturbance; and estimating the edge probability of salient objects to boost the detail of decoding results. The proposed method MEPNet is mainly composed of four kinds of modules, including Lite Receptive Filed Block (RFB-Lite) module, Prior Query (PQ) module, Full-scale sub-Decoder (FD) module, and Edge Auxiliary (EA) module. The RFB-Lite adopts the multi-scale convolution with grouped branches to efficiently reduce channel redundancy and enhance the diversity of semantics. The PQ introduces the mask prior-distribution into the fused feature by multi-head cross-attention. To solve the conflict between the number of attention heads and the computational complexity in cross-attention, Multi-head L-Cross Attention (MLAC) is proposed to self-weight the feature while calculating the attention score matrix and global attention. The FD realizes the bi-direction decoding with feature pyramid network (PFN) and reversed feature pyramid network (FPN-R). Three FDs adopted in MEPNet present a multi-basis decoding to fully utilize multi-scale features. The alternating use of PQ and FD ensures the suppression of background disturbance. The EA, as the last part of MEPNet, boosts the decoding results with edge filter estimating the edge probability and edge refine correcting the up-sampled decoding results. The SOD experimental results on DUTS-TE, HKU-IS, PASCAL-S, ECSSD, and DUT-OMRON datasets demonstrate that the proposed MEPNet is more robust under different complex scenes when compared to some state-of-the-art (SOTA) methods.


## Prerequisites
- [Python 3.6](https://www.python.org/)
- [Pytorch 1.10](http://pytorch.org/)
- [OpenCV 4.5.5.64](https://opencv.org/)
- [Numpy 1.19.5](https://numpy.org/)
- [pillow 8.4.0](https://pypi.org/project/Pillow/)
- [timm 0.4.12](https://pypi.org/project/timm/0.4.12/)
- [tqdm 4.64.0](https://pypi.org/project/tqdm/4.64.0/)


## Clone repository

```shell
git clone https://github.com/ZEROICEWANG/MEPNet.git
cd MEPNet/
```

## Download dataset

Download the following datasets and unzip them into `../SOD_Data` folder

- [PASCAL-S](http://cbi.gatech.edu/salobj/)
- [ECSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html)
- [HKU-IS](https://i.cs.hku.hk/~gbli/deep_saliency.html)
- [DUT-OMRON](http://saliencydetection.net/dut-omron/)
- [DUTS](http://saliencydetection.net/duts/)


## Download model

- If you want to test the performance of MEPNet, please download the model([Baidu](https://pan.baidu.com/s/1S_uwKUEUIoRMw-p9Ek28zA?pwd=6jyz) [Google](https://drive.google.com/file/d/1-2gtqk9M3Ex9Ou_YtOSFsQmvPWkcOySg/view?usp=sharing)) into `models/RES_Model` folder, and download the pretrained model ([Baidu](https://pan.baidu.com/s/1Lh-MrKSLU1rG6DL45PiqtQ?pwd=pfvi) [Google](https://drive.google.com/file/d/1Y8d2cSZh71oKd4qQ56TTK_sYvhTIHe8G/view?usp=sharing)) into `models/pretrained/prior_query2_channel_last_L_attention_rm_SA/pretrain` folder.


## Training

```shell
    python3 train.py # or using bash cmd_train.sh
```


## Testing

```shell
    python3 predict.py # or using bash cmd.sh
```
- After testing, saliency maps of `PASCAL-S`, `ECSSD`, `HKU-IS`, `DUT-OMRON`, `DUTS-TE` will be saved in `predict_result/` folder.

## Saliency maps & Pre-Trained model & Trained model
- saliency maps: [Baidu](https://pan.baidu.com/s/1EEqGAK5KU-Frpsvx9GKFDw?pwd=by1m) [Google](https://drive.google.com/file/d/16WoEBpne1mpsa_NEQ1ffFG9v7ZnppqxR/view?usp=sharing)

- pretrained model: [Baidu](https://pan.baidu.com/s/1Lh-MrKSLU1rG6DL45PiqtQ?pwd=pfvi) [Google](https://drive.google.com/file/d/1Y8d2cSZh71oKd4qQ56TTK_sYvhTIHe8G/view?usp=sharing)

- trained model: [Baidu](https://pan.baidu.com/s/1S_uwKUEUIoRMw-p9Ek28zA?pwd=6jyz) [Google](https://drive.google.com/file/d/1-2gtqk9M3Ex9Ou_YtOSFsQmvPWkcOySg/view?usp=sharing)

full source code will be realised soon

