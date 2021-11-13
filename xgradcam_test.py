#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 18:55:58 2021

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

from utils import visualize
from cam_techniques import XGradCAM

    
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

imgTransform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

pathImg = 'images/ILSVRC2012_val_00000073.JPEG'

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

vgg = models.vgg16(pretrained=True).eval().to(device)
layer = vgg.features[29]

xgradcamVGG = XGradCAM(vgg, layer)

imgPIL = Image.open(pathImg)
img = imgTransform(imgPIL).to(device).unsqueeze(0)

since = time.time()
xgradcamMap, prediction = xgradcamVGG(img.to(device))
print(f'Time taken {time.time()-since}')

visualize(np.array(imgPIL), xgradcamMap.cpu().numpy(), save_path='xgradcam_op.png')
