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

from scorecam import ScoreCAM

from PIL import Image

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
    
imgTransform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])


pathImg = '../images/ILSVRC2012_val_00000073.JPEG'

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

vgg = models.vgg16(pretrained=True).eval().to(device)
layer = vgg.features[29]

scorecamVGG = ScoreCAM(vgg, layer)

imgPIL = Image.open(pathImg)
img = imgTransform(imgPIL).unsqueeze(0)
    
scorecamMap, prediction = scorecamVGG(img.to(device))

plt.imshow(scorecamMap[0].squeeze().cpu())
plt.show()

plt.imshow(imgPIL)
plt.show()

print(prediction)











