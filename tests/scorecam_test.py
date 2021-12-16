#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 15:30:59 2021

@author: sagar
"""

import torch
import torchvision.models as models
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import time
import os
import numpy as np

from utils import visualize, map_class
from cam_techniques import ScoreCAM
from PIL import Image

imagenet_class_file = 'misc/map_clsloc.txt'
model_class_file = 'misc/imagenet1000_clsidx_to_labels.txt'

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
    
imgTransform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])


pathImgs = ['/mnt/nas/share/sagar/XAI/val_categorised/590/ILSVRC2012_val_00006215.JPEG',
           '/mnt/nas/share/sagar/XAI/val_categorised/591/ILSVRC2012_val_00006227.JPEG',
           '/mnt/nas/share/sagar/XAI/val_categorised/596/ILSVRC2012_val_00004802.JPEG',
           '/mnt/nas/share/sagar/XAI/val_categorised/667/ILSVRC2012_val_00009855.JPEG',
           '/mnt/nas/share/sagar/XAI/val_categorised/689/ILSVRC2012_val_00019027.JPEG',
           '/mnt/nas/share/sagar/XAI/val_categorised/764/ILSVRC2012_val_00011982.JPEG',
           '/mnt/nas/share/sagar/XAI/val_categorised/887/ILSVRC2012_val_00001805.JPEG',
           '/mnt/nas/share/sagar/XAI/val_categorised/1000/ILSVRC2012_val_00004949.JPEG']


device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

vgg = models.vgg16(pretrained=True).eval().to(device)
layer = vgg.features[29]

scorecamVGG = ScoreCAM(vgg, layer)

since = time.time()  

for pathImg in pathImgs:
    model_classid, class_name = map_class(int(pathImg.split('/')[-2]), imagenet_class_file, model_class_file) 
    imgPIL = Image.open(pathImg)
    img = imgTransform(imgPIL).unsqueeze(0)
    
    scorecamMap, prediction = scorecamVGG(img.to(device))
    
    if not os.path.exists(os.path.join('outputs', class_name)):
        os.makedirs(os.path.join('outputs',class_name))
    save_path = os.path.join('outputs', class_name, 'score_'+pathImg.split('/')[-1])  
    visualize(np.array(transforms.Resize(224)(imgPIL)), scorecamMap.cpu().numpy(), save_path)


print(f'Time taken {time.time()-since}')





