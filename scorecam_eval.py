#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 19:08:44 2021

@author: sagar
"""

import os
import time
import torch

import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
from cam_techniques import ScoreCAM
from utils import map_class, MetricsMeter, get_explanation_map


EVALUATE_LR = True
EVALUATE_ALL = False # Flag. Evaluate only on correct predictions if False 


datasetPath = '/mnt/nas/share/sagar/XAI/val_categorised'
imagenet_class_file = 'misc/map_clsloc.txt'
model_class_file = 'misc/imagenet1000_clsidx_to_labels.txt'

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
   
classIDs = ['590', '591', '596', '667', '689', '764', '887', '1000']
 
imgTransform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])


device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

vgg = models.vgg16(pretrained=True).eval().to(device)
layer = vgg.features[29]

scorecamVGG = ScoreCAM(vgg, layer)
metrics = MetricsMeter()

num_samples = 0
num_corrects = 0
total = 0
for classid in classIDs if classIDs!=-1 else os.listdir(datasetPath):
    model_classid, class_name = map_class(int(classid), imagenet_class_file, model_class_file)
    print(f'Processing class {classid} {class_name}')
    for filename in os.listdir(os.path.join(datasetPath, classid)):
        if filename.endswith('JPEG'):
            total+=1
            pathImg = os.path.join(datasetPath, classid, filename)
            imgPIL = Image.open(pathImg)
            
            if EVALUATE_LR:
                imgPIL = transforms.Resize(100)(imgPIL)

            if len(np.array(imgPIL).shape) == 3:
                img = imgTransform(imgPIL).unsqueeze(0).to(device)
                ablationcamMap, logits = scorecamVGG(img)

                pred_class = torch.argmax(logits)

                if pred_class == model_classid:
                    num_corrects+=1
                    
                if pred_class == model_classid or EVALUATE_ALL:
                    explanationMap = get_explanation_map(ablationcamMap)
                    imgNew = img*explanationMap.unsqueeze(0)
                    with torch.no_grad():
                        logitsNew = vgg(imgNew).squeeze()
                    metrics.update(logits.cpu(), logitsNew.cpu(), model_classid)
                    num_samples += 1
        
metrics.average(num_samples)       
 
print(round(metrics.drop_in_conf*100, 2))       
print(round(metrics.drop_in_act*100, 2))       
print(round(metrics.inc_in_conf*100, 2))       
print(round(metrics.inc_in_act*100, 2))       

print('Accuracy: {}'.format(num_corrects/total)) 
