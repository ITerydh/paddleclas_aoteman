# 基于PaddleClas2.2进行奥特曼分类，利用Paddle-Lite进行安卓部署！

项目开源地址：
> [https://aistudio.baidu.com/aistudio/projectdetail/2219455](https://aistudio.baidu.com/aistudio/projectdetail/2219455)

- [基于PaddleClas2.2进行奥特曼分类，利用Paddle-Lite进行安卓部署！](#--paddleclas22----------paddle-lite-------)
  * [0 项目背景](#0-----)
  * [1 项目如何实现？](#1--------)
  * [2 项目数据集介绍](#2--------)
  * [3 数据处理](#3-----)
    + [3.1 读取数据](#31-----)
    + [3.2 打乱数据](#32-----)
    + [3.3 划分数据](#33-----)
    + [3.4 数据预处理](#34------)
  * [4 采用paddleclas进行训练](#4---paddleclas----)
    + [4.1 修改配置文件](#41-------)
    + [4.2 添加类别映射文件](#42---------)
  * [5 模型预测](#5-----)
    + [5.1 验证](#51---)
    + [5.2 再次验证](#52-----)
  * [6 基于Paddle-Lite进行安卓部署](#6---paddle-lite------)
    + [6.1 下载PaddleLite-android-demo工程包](#61---paddlelite-android-demo---)
    + [6.2 官方demo安卓部署运行](#62---demo------)
    + [6.3 将官方demo改为我们的模型进行部署](#63----demo-----------)
      - [1 导出预测模型](#1-------)
      - [2 利用paddlelite将模型转为.nb文件](#2---paddlelite-----nb--)
      - [3 下载.nb文件备用](#3---nb----)
      - [4 更改各种文件，进行部署](#4------------)
      - [5 成功运行](#5-----)
- [总结](#--)
  * [项目总结](#----)
  * [个人总结](#----)

![](https://ai-studio-static-online.cdn.bcebos.com/8da85bfb5d7b4ee0885004575db9d7912621a81cd5594ddb965f90b247cea996)


> 注意：整个项目有录制视频进行详细介绍：

B站搜索用户：iterhui


## 0 项目背景
<center><strong>新的风暴已经出现，怎么能够停滞不前~</strong></center>  
<hr>
<center>本人亲情之作，花费了不少的时间去完成了现在的数据集</center> 
<center>包含<strong>四类</strong>奥特曼:<font color="#FF0000">迪迦、杰克、赛文和泰罗奥特曼。</font></center>  
<center>唤醒你的童年~</center> 
<center>女朋友还不认识奥特曼是什么？</center>   
<center>甩给她！（当然后续搞个口红色号分类？？也不是不可QAQ）</center> 
<center>虽然只有四类奥特曼，但聪明的你自行补充完善这个项目我猜没什么毛病吧~</center> 
<center>其实想偷懒，截图截得手麻了,中途试了用爬虫，感觉不带劲</center>


## 1 项目如何实现？
采用paddleclas进行图像分类任务。  
**paddleclas**官方文档连接如下：  
>[https://gitee.com/paddlepaddle/PaddleClas/blob/release/2.2/docs/zh_CN/tutorials/quick_start_new_user.md](https://gitee.com/paddlepaddle/PaddleClas/blob/release/2.2/docs/zh_CN/tutorials/quick_start_new_user.md)


## 2 项目数据集介绍
包含四类奥特曼，迪迦200张、杰克100张、赛文100张和泰罗奥特曼150张。  
！！！数据集完全由我一张一张截图而来，切勿作为其他用途，仅供个人学习！！！  
<strong>友情提示：数据集仅供学习和个人使用，如果被告我不负责（孩子怕极了）</strong>

```
├── aoteman  
│   ├── dijia  
│   │   ├── 001.jpg  
│   │   ├── 002.jpg  
│   │   ├── 003.jpg  
│   │   ├── ......  
│   │   ├── 198.jpg
│   │   ├── 199.jpg
│   │   └── 200.jpg
│   ├── jieke
│   │   ├── 001.jpg
│   │   ├── 002.jpg
│   │   ├── 003.jpg
│   │   ├── ......
│   │   ├── 098.jpg
│   │   ├── 099.jpg
│   │   └── 100.jpg
│   ├── saiwen
│   │   ├── 001.jpg
│   │   ├── 002.jpg
│   │   ├── 003.jpg
│   │   ├── ......
│   │   ├── 098.jpg
│   │   ├── 099.jpg
│   │   └── 100.jpg
│   ├── tailuo
│   │   ├── 001.jpg
│   │   ├── 002.jpg
│   │   ├── 003.jpg
│   │   ├── ......
│   │   ├── 148.jpg
│   │   ├── 149.jpg
│   │   └── 150.jpg
```

**话不多说，开整！**



```python
!python3 -c "import paddle; print(paddle.__version__)"
```

    2.1.0



```python
# 解压数据集
!unzip -oq /home/aistudio/data/data101651/aoteman.zip
```


```python
# 更清楚的看文件的结构
!tree
```


```python
# 安装paddleclas以及相关三方包(好像studio自带的已经够用了，无需安装了)
!git clone https://gitee.com/paddlepaddle/PaddleClas.git -b release/2.2
# 我这里安装相关包时，花了30几分钟还有错误提示，不管他即可
#!pip install --upgrade -r PaddleClas/requirements.txt -i https://mirror.baidu.com/pypi/simple
```

    Cloning into 'PaddleClas'...
    remote: Enumerating objects: 538, done.[K
    remote: Counting objects: 100% (538/538), done.[K
    remote: Compressing objects: 100% (323/323), done.[K
    remote: Total 15290 (delta 344), reused 349 (delta 210), pack-reused 14752[K
    Receiving objects: 100% (15290/15290), 113.56 MiB | 12.55 MiB/s, done.
    Resolving deltas: 100% (10236/10236), done.
    Checking connectivity... done.



```python
# 查看都安装上了没
!pip list package
```


```python
# 忽略（垃圾）警告信息
# 在python中运行代码经常会遇到的情况是——代码可以正常运行但是会提示警告，有时特别讨厌。
# 那么如何来控制警告输出呢？其实很简单，python通过调用warnings模块中定义的warn()函数来发出警告。我们可以通过警告过滤器进行控制是否发出警告消息。
import warnings
warnings.filterwarnings("ignore")
```

## 3 数据处理
  (4数据预处理这个在paddleclas中进行处理了)


```python
# 导入所需要的库
from sklearn.utils import shuffle
import os
import pandas as pd
import numpy as np
from PIL import Image
import paddle
import paddle.nn as nn
from paddle.io import Dataset
import paddle.vision.transforms as T
import paddle.nn.functional as F
from paddle.metric import Accuracy
import random
```

### 3.1 读取数据


```python
# -*- coding: utf-8 -*-
# 根据官方paddleclas的提示，我们需要把图像变为两个txt文件
# 我们总共是200+100+100+150=550张图片，按照经典的划分方式0.8：0.2
# train_list.txt（训练集，440张图）
# val_list.txt（验证集，110张图）
# 先把路径搞定 比如：dataset/dijia/001.png ,读取到并写入txt 
#                  dataset/jieke/001.png                 
#                  dataset/saiwen/001.png                
#                  dataset/tailuo/001.png  

dirpath = "aoteman"
# 先得到总的txt后续再进行划分，因为要划分出验证集，所以要先打乱，因为原本是有序的
def get_all_txt():
    all_list = []
    i = 0
    for root,dirs,files in os.walk(dirpath): # 分别代表根目录、文件夹、文件
        for file in files:
            i = i + 1 
            # 文件中每行格式： 图像相对路径      图像的label_id（注意：中间有空格）。              
            #                aoteman/dijia/001.png    0
            #                aoteman/jike/001.png     1
            #                aoteman/saiwen/001.png   2
            #                aoteman/tailuo/001.png   3
            if("dijia" in root):
                all_list.append(os.path.join(root,file)+" 0\n")
            if("jieke" in root):
                all_list.append(os.path.join(root,file)+" 1\n")
            if("saiwen" in root):
                all_list.append(os.path.join(root,file)+" 2\n")
            if("tailuo" in root):
                all_list.append(os.path.join(root,file)+" 3\n")
    allstr = ''.join(all_list)
    f = open('all_list.txt','w',encoding='utf-8')
    f.write(allstr)
    return all_list , i

all_list,all_lenth = get_all_txt()
print(all_lenth-1) # 有意者是预测的图片，得减去
```

    550


### 3.2 打乱数据


```python
# 思路 ： 先把数据打乱，然后按照比例划分数据集
random.shuffle(all_list)
random.shuffle(all_list)
```

### 3.3 划分数据


```python
# 我们总共是200+100+100+150=550张图片，按照经典的划分方式0.8：0.2
# train_list.txt（训练集，440张图）
# val_list.txt（验证集，110张图）

train_size = int(all_lenth * 0.8)
train_list = all_list[:train_size]
val_list = all_list[train_size:]

print(len(train_list))
print(len(val_list))
```

    440
    110



```python
# 运行cell，生成txt 
train_txt = ''.join(train_list)
f_train = open('train_list.txt','w',encoding='utf-8')
f_train.write(train_txt)
f_train.close()
print("train_list.txt 生成成功！")
```

    train_list.txt 生成成功！



```python
# 运行cell，生成txt
val_txt = ''.join(val_list)
f_val = open('val_list.txt','w',encoding='utf-8')
f_val.write(val_txt)
f_val.close()
print("val_list.txt 生成成功！")
```

    val_list.txt 生成成功！


### 3.4 数据预处理

这里没有进行数据预处理，因为有了PaddleClas之后，数据预处理直接在yaml文件进行更改即可，后续**4.1**会提到。


```python
# 将图片移动到paddleclas下面的数据集里面
# 至于为什么现在移动，也是我的一点小技巧，防止之前移动的话，生成的txt的路径是全路径，反而需要去掉路径的一部分
!mv aoteman/ PaddleClas/dataset/
!mv all_list.txt PaddleClas/dataset/aoteman
!mv train_list.txt PaddleClas/dataset/aoteman
!mv val_list.txt PaddleClas/dataset/aoteman
```

## 4 采用paddleclas进行训练

数据集核实完搞定成功的前提下，可以准备更改原文档的参数进行实现自己的图片分类了！


```python
#windows在cmd中进入PaddleClas根目录，执行此命令
%cd PaddleClas
!ls
```

    /home/aistudio/PaddleClas
    dataset  hubconf.py   MANIFEST.in    README_ch.md  requirements.txt
    deploy	 __init__.py  paddleclas.py  README_en.md  setup.py
    docs	 LICENSE      ppcls	     README.md	   tools


### 4.1 修改配置文件

主要是以下几点：分类数、图片总量、训练和验证的路径、图像尺寸、**数据预处理**、训练和预测的num_workers: 0才可以在aistudio跑通。


> PaddleClas/ppcls/configs/quick_start/new_user/ShuffleNetV2_x0_25.yaml
```
# global configs
Global:
  checkpoints: null
  pretrained_model: null
  device: gpu
  output_dir: ./output/
  save_interval: 20
  eval_during_train: True
  eval_interval: 10
  epochs: 600
  print_batch_step: 10
  use_visualdl: True
  # used for static mode and model export
  image_shape: [3, 224, 224]
  save_inference_dir: ./inference

# model architecture
Arch:
  name: ShuffleNetV2_x0_25
  class_num: 4
 
# loss function config for traing/eval process
Loss:
  Train:
    - CELoss:
        weight: 1.0
  Eval:
    - CELoss:
        weight: 1.0


Optimizer:
  name: Momentum
  momentum: 0.9
  lr:
    name: Cosine
    learning_rate: 0.0125
    warmup_epoch: 5
  regularizer:
    name: 'L2'
    coeff: 0.00001


# data loader for train and eval
DataLoader:
  Train:
    dataset:
      name: ImageNetDataset
      image_root: ./dataset/
      cls_label_path: ./dataset/aoteman/train_list.txt
      transform_ops:
        - DecodeImage:
            to_rgb: True
            channel_first: False
        - ResizeImage:
            resize_short: 256
        - CropImage:
            size: 224
        - RandFlipImage:
            flip_code: 1
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''

    sampler:
      name: DistributedBatchSampler
      batch_size: 16
      drop_last: False
      shuffle: True
    loader:
      num_workers: 0
      use_shared_memory: True

  Eval:
    dataset: 
      name: ImageNetDataset
      image_root: ./dataset/
      cls_label_path: ./dataset/aoteman/val_list.txt
      transform_ops:
        - DecodeImage:
            to_rgb: True
            channel_first: False
        - ResizeImage:
            resize_short: 256
        - CropImage:
            size: 224
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''
    sampler:
      name: DistributedBatchSampler
      batch_size: 64
      drop_last: False
      shuffle: False
    loader:
      num_workers: 0
      use_shared_memory: True

Infer:
  infer_imgs: dataset/aoteman/predict_demo.jpg
  batch_size: 10
  transforms:
    - DecodeImage:
        to_rgb: True
        channel_first: False
    - ResizeImage:
        resize_short: 256
    - CropImage:
        size: 224
    - NormalizeImage:
        scale: 1.0/255.0
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
        order: ''
    - ToCHWImage:
  PostProcess:
    name: Topk
    topk: 4
    class_id_map_file: ppcls/configs/quick_start/new_user/aoteman_label_list.txt

Metric:
  Train:
    - TopkAcc:
        topk: [1, 4]
  Eval:
    - TopkAcc:
        topk: [1, 4]

```

### 4.2 添加类别映射文件
> PaddleClas/ppcls/configs/quick_start/new_user/aoteman_label_list.txt
```
0 迪迦奥特曼  
1 杰克奥特曼  
2 赛文奥特曼  
3 泰罗奥特曼
```


```python
!export CUDA_VISIBLE_DEVICES=0
```


```python
# 开始训练 
!python tools/train.py \
    -c ./ppcls/configs/quick_start/new_user/ShuffleNetV2_x0_25.yaml
```

或许因为奥特曼之间的区别还是挺大的，最后的结果基本上都接近1了！

![](https://ai-studio-static-online.cdn.bcebos.com/f5672233bed144d0bd898e4678d5694f8438a3041d8645b583f8ebe8e17d521a)


## 5 模型预测

### 5.1 验证


```python
!python3 tools/infer.py \
    -c ./ppcls/configs/quick_start/new_user/ShuffleNetV2_x0_25.yaml \
    -o Infer.infer_imgs=dataset/aoteman/predict_demo.jpg \
    -o Global.pretrained_model=output/ShuffleNetV2_x0_25/latest
```

真实的图片是：   
![迪迦奥特曼](https://ai-studio-static-online.cdn.bcebos.com/d40837913cc74354a4c19e3f6bdf43fae130ea855d0e416e8a5822a08c30031e) 


预测的结果是：   
>'class_ids': [0, 3, 1, 2], 'scores': [0.99999, 1e-05, 0.0, 0.0]，'label_names': ['迪迦奥特曼', '泰罗奥特曼', '杰克奥特曼', '赛文奥特曼']  

也就是说0的概率最大，0对应的结果是迪迦，也就是说结果为迪迦，预测无误。   

### 5.2 再次验证


```python
# 再来一张其他试试，防止有意外情况，自行百度找图，在下面jpg替换即可
!python3 tools/infer.py \
    -c ./ppcls/configs/quick_start/new_user/ShuffleNetV2_x0_25.yaml \
    -o Infer.infer_imgs=dataset/aoteman/predict_tailuo.jpg \
    -o Global.pretrained_model=output/ShuffleNetV2_x0_25/latest
```

真实的图片是：   
![泰罗奥特曼](https://ai-studio-static-online.cdn.bcebos.com/4096b6b3572d4268897c71c2fd85ff0db63c92f467074307b5944c8ce09215c0)


预测的结果是：   
>'class_ids': [3, 2, 1, 0], 'scores': [0.98939, 0.01061, 0.0, 0.0]，'label_names': ['泰罗奥特曼', '赛文奥特曼', '杰克奥特曼', '迪迦奥特曼'] 

也就是说3的概率为0.98939，最大，3对应的结果是泰罗，也就是说结果为泰罗，预测无误。 

## 6 基于Paddle-Lite进行安卓部署

Paddle-Lite 是飞桨推出的一套功能完善、易用性强且性能卓越的轻量化推理引擎。 轻量化体现在使用较少比特数用于表示神经网络的权重和激活，能够大大降低模型的体积，解决终端设备存储空间有限的问题，推理性能也整体优于其他框架。 PaddleClas 使用 Paddle-Lite 进行了移动端模型的性能评估，本部分以本案例的奥特曼数据集的ShuffleNetV2_x0_25模型为例，介绍怎样使用Paddle-Lite，在移动端(基于骁龙855的安卓开发平台)对进行模型速度评估。

> 部署参考文档：  
> [https://paddleclas.readthedocs.io/zh_CN/latest/extension/paddle_mobile_inference.html](https://paddleclas.readthedocs.io/zh_CN/latest/extension/paddle_mobile_inference.html)  
> [https://paddle-lite.readthedocs.io/zh/latest/demo_guides/android_app_demo.html](https://paddle-lite.readthedocs.io/zh/latest/demo_guides/android_app_demo.html)

1. 目的：将基于Paddle-Lite预测库的Android APP 部署到手机，实现图像分类

2. 需要的环境： Android Studio、Android手机（开启USB调试模式）、下载到本地的[Paddle-Lite-Demo](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/master/PaddleLite-android-demo)工程

3. 预先要求：如果您的Android Studio尚未配置NDK，请根据Android Studio用户指南中的安装及配置NDK和CMake内容，预先配置好NDK。您可以选择最新的NDK版本，或者与Android编译环境配置中的NDK版本保持一致。

### 6.1 下载PaddleLite-android-demo工程包

这里先不讲解推理预测的模型，直接先拿官方的这个工程包运行，看是否管用再说。

![](https://ai-studio-static-online.cdn.bcebos.com/439918eba99a4035828b14fb28304fc62f42eea262504e828013b0623988c0cb)


将下载的模型包解压

![](https://ai-studio-static-online.cdn.bcebos.com/df8bb878d86347638ff27675284df318d56b046c12c24acea00469007fed7ed5)

只留下 **image_classification_demo**即可  

![](https://ai-studio-static-online.cdn.bcebos.com/4016f92fcea243f39e3eaf4cd8714c4992a326bc4e2247df94084d8ad5d8929a)

![](https://ai-studio-static-online.cdn.bcebos.com/8445c8e7610f47dd88328b1435e74cef38b8060f747c43f2b5b55f0c748ad0f5)






### 6.2 官方demo安卓部署运行

android studio软件安装详情查看我的csdn：   
[Android Studio的安装，史上最详细(超多图)！！](https://blog.csdn.net/qq_41976613/article/details/91432304)  
[android studio的安装（补充篇gradle失败的问题）](https://blog.csdn.net/qq_41976613/article/details/104394870)  

1. 导入已经存在的包   
  ![](https://ai-studio-static-online.cdn.bcebos.com/4a9cabd34f30475bbd6c8d8c810093c021f64347730b4fe5a18836f0f4941551)

2. 更新gradle依赖库  
  ![](https://ai-studio-static-online.cdn.bcebos.com/2a1779d650fd4ce7828bbae2f241481ad341ba5844b5499491a2e473c296e520)

3. 连接手机（这里我使用夜神模拟器连接安卓，上述补充篇有提到夜神模拟器如何安装操作）   
  （不建议使用As自带的安卓，不好联网）  
  逐步按照操作来即可  
  ![](https://ai-studio-static-online.cdn.bcebos.com/892c3f6ad8794762b6f6b6b37f84a879e6f5abe0edf3446bbe1cba8949d6648f)   
  ![](https://ai-studio-static-online.cdn.bcebos.com/998f930df0224da18ae382ccffb3495b73dd9483415440b3bb58c3957bc22fee)    
  ![](https://ai-studio-static-online.cdn.bcebos.com/d8648b6b7db845bdaa3ff82abe40522efca2f06f7928409abb0a8b4e9b411886)    
  ![](https://ai-studio-static-online.cdn.bcebos.com/ec22fec05dc543d186d444c404b81d045ec47062ed6f45c596591fd2014a9a01)    

4.看到上图证明已经全部ok。点击上图右侧的启动按钮，执行以下看官方demo是否可用!

等待加载......    
![](https://ai-studio-static-online.cdn.bcebos.com/0f0568c44598401cbaa5e9c09d0e8b1cbaa5131e63c5495683903d07899c2158)  
![](https://ai-studio-static-online.cdn.bcebos.com/6259c6b0e1ee45978ae1243f49b464d3bb5d78b60d8343aca811c22d3da01034)   

5. 走到这一步，说明官方demo可用   
  如果想使用官方的demo，只需要右上角的三个点，进行拍照或者读取图片即可，自行尝试不再演示。

### 6.3 将官方demo改为我们的模型进行部署


```python
#windows在cmd中进入PaddleClas根目录，执行此命令
%cd PaddleClas
!ls
```

    /home/aistudio/PaddleClas
    dataset  hubconf.py   LICENSE	   paddleclas.py  README_en.md	    setup.py
    deploy	 inference    MANIFEST.in  ppcls	  README.md	    tools
    docs	 __init__.py  output	   README_ch.md   requirements.txt


#### 1 导出预测模型


```python
# 导出模型
!python3 tools/export_model.py \
    -c ppcls/configs/quick_start/new_user/ShuffleNetV2_x0_25.yaml \
    -o Global.pretrained_model=output/ShuffleNetV2_x0_25/latest
```

#### 2 利用paddlelite将模型转为.nb文件


```python
# 准备PaddleLite依赖
!pip install paddlelite
```

    Looking in indexes: https://mirror.baidu.com/pypi/simple/
    Collecting paddlelite
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/77/5e/2f463252bcf7de6d594cff2a89f76c643088946203846fbdb845f9cb5378/paddlelite-2.9-cp37-cp37m-manylinux1_x86_64.whl (45.8MB)
    [K     |████████████████████████████████| 45.8MB 8.9MB/s eta 0:00:013
    [?25hInstalling collected packages: paddlelite
    Successfully installed paddlelite-2.9



```python
# 准备PaddleLite部署模型
#--valid_targets中参数（arm）用于传统手机，（npu,arm ）用于华为带有npu处理器的手机
!paddle_lite_opt \
    --model_file=inference/inference.pdmodel \
    --param_file=inference/inference.pdiparams \
    --optimize_out=./inference/ShuffleNetV2_x0_25 \
    --optimize_out_type=naive_buffer \
    --valid_targets=arm 
    #--valid_targets=npu,arm 
```

    Loading topology data from inference/inference.pdmodel
    Loading params data from inference/inference.pdiparams
    1. Model is successfully loaded!
    2. Model is optimized and saved into ./inference/ShuffleNetV2_x0_25.nb successfully


#### 3 下载.nb文件备用

路径 PaddleClas/inference/ShuffleNetV2_x0_25.nb

3.1 打开下载的官方demo文件，如下图所在位置新建文件夹 shufflenet_v2

![](https://ai-studio-static-online.cdn.bcebos.com/fc62791c5fe848e596b757112c0cc1e09b650affb8e740e18be4f8b30ea3211f)     

3.2 ShuffleNetV2_x0_25.nb将文件放入进去，并改名为 model.nb   
![](https://ai-studio-static-online.cdn.bcebos.com/9edc77ce4677499486390d5412252f9278d4c4d582e04a84b22fd451a97e72ee)   

3.3 放入label标签文件   
路径为ppcls/configs/quick_start/new_user/aoteman_label_list.txt   
直接下载了，放在对应文件夹下即可   
![](https://ai-studio-static-online.cdn.bcebos.com/cdb2133ed4b0435ca20bae78577813d258c2051622f14daf915caf09f1ce2406)

3.4 放入预测文件图片（自行百度一张扔进去即可），也就是预测demo，也就是进入界面的默认预测图片

![](https://ai-studio-static-online.cdn.bcebos.com/6ffe0f267141445a8263b6e77d143ec6f2dc8c7e886c4f44a3c0258675523ada)



#### 4 更改各种文件，进行部署

就是我们刚刚修改的三个文件的地方  
```
<string name="MODEL_PATH_DEFAULT">models/shufflenet_v2</string>
<string name="LABEL_PATH_DEFAULT">labels/aoteman_label_list.txt</string>
<string name="IMAGE_PATH_DEFAULT">images/pro.png</string>
```

![](https://ai-studio-static-online.cdn.bcebos.com/5d0568cece144bfdb751b5dfea3152c230767456335c4764ade199b2d78b5146)


#### 5 成功运行

可以成功看到预测结果的画面 

![](https://ai-studio-static-online.cdn.bcebos.com/e8fb3f0feefb40c2b9438f4f90061341018394d991c24b8ba36a68d78df9f98e)
![](https://ai-studio-static-online.cdn.bcebos.com/d1aaf22fd770422ba099a8668cc1eb004a748662389a448086d88efd868df942)



此外：
右上角有三个点，依次为：
1. 照片预测
2. 相机预测
3. 设置


# 总结
## 项目总结
1. 使用下来，用了很多版本的paddleclas，比如2.1,2.2，develop最后还是选择使用了2.2
2. 版本差异如下：  
   2.1 生成的模型是文件夹存储的形式，并且有最佳模型文件  
   2.2 生成的模型文件直接排序在一个大文件夹下、支持写一个预测类别文件，预测输出时直接可以对照看是哪个类别。  
3. 使用paddleclas不管是哪个版本，最主要的还是数据处理和调参  
   3.1 数据处理，将信息转变为txt：相对路径+空格+类别  
   3.2 调参，变成自己的对应信息  
   主要是以下几点：分类数、图片总量、训练和验证的路径、图像尺寸、训练和预测的num_workers: 0才可以在aistudio跑通。  
4. 安卓部署版本已完成，冲冲冲！


## 个人总结
>全网同名: iterhui

我在AI Studio上获得钻石等级，点亮10个徽章，来互关呀~

>[https://aistudio.baidu.com/aistudio/personalcenter/thirdview/643467](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/643467)

