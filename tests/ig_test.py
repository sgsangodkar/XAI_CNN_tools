#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 18:34:11 2021

@author: sagar
"""
import cv2
import torch
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt
import numpy as np
import time
import os

from utils import visualize, visualize_multiscale, map_class
from cam_techniques import IG

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

igVGG = IG(vgg, layer)


def reverse_transform(img):
    img = img.transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)

    return img

for pathImg in pathImgs:
    imagenet_id = int(pathImg.split('/')[-2])
    model_classid, class_name = map_class(imagenet_id, imagenet_class_file, model_class_file)
    imgPIL = Image.open(pathImg)
    LRimgPIL = transforms.Resize(100)(imgPIL)
    img = imgTransform(imgPIL).to(device).unsqueeze(0)
    LRimg = imgTransform(LRimgPIL).to(device).unsqueeze(0)
    
    since = time.time()
    igMap = igVGG(img.to(device)).cpu().numpy()
    igMap = (igMap-np.min(igMap))/(np.max(igMap)-np.min(igMap))
    igMap = (igMap*255).astype(np.uint8)
    
    LRigMap = igVGG(LRimg.to(device)).cpu().numpy()
    LRigMap = (LRigMap-np.min(LRigMap))/(np.max(LRigMap)-np.min(LRigMap))
    LRigMap = (LRigMap*255).astype(np.uint8)

    
    if not os.path.exists(os.path.join('outputs', class_name)):
        os.makedirs(os.path.join('outputs',class_name))
        
    img = reverse_transform(img.squeeze().cpu().numpy())    
    LRimg = reverse_transform(LRimg.squeeze().cpu().numpy())    

    save_path = os.path.join('outputs', class_name, 'ig'+pathImg.split('/')[-1])  
    
    fig, ax = plt.subplots(2,2)
    ax[0,0].imshow(img, 'gray')
    ax[0,1].imshow(LRimg, 'gray')
    ax[1,0].imshow(5*igMap, 'gray')
    ax[1,1].imshow(5*LRigMap, 'gray')
    fig.show()
    #visualize_ig_multiscale(img, LRimg, igMap, LRigMap, save_path)
    
print(f'Time taken {time.time()-since}')
