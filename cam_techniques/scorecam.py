#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 15:40:40 2021

@author: sagar
"""

import torch
import torch.nn.functional as F

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'


class ScoreCAM(object):
    def __init__(self, model, layer):
        self.model = model
        self.layer = layer
        self.activations = {}

        def forward_hook(module, input, output):
            self.activations['activations'] = output.to(device)
            return None
        
        self.layer.register_forward_hook(forward_hook)
        
    def __call__(self, input, classid=None):
        logits = self.model(input).squeeze()
        _, _, ht, wt = input.shape
        
        if classid:
            imgClass = torch.LongTensor([classid])
        else:
            imgClass = torch.argmax(logits)
            

        activations = self.activations['activations']
        _, C, u, v = activations.shape
        fused_saliency_map= torch.zeros((1,1,ht,wt)).to(device)

        with torch.no_grad():
          for i in range(C):
              # upsampling
              saliency_map = activations[:, i:i+1, :, :]
              saliency_map = F.interpolate(saliency_map, size=(ht, wt), mode='bilinear', align_corners=False)
              
              if saliency_map.max() != saliency_map.min():
                  norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
                 
              output1 = self.model(input * norm_saliency_map)
              output2 = self.model(torch.zeros((1,3,ht,wt)).to(device))
              output = F.softmax(output1-output2, dim=1).squeeze()
              alpha = output[imgClass]
              
              
              fused_saliency_map +=  (alpha) * saliency_map
                
        fused_saliency_map = F.relu(fused_saliency_map)

        if fused_saliency_map.max() != fused_saliency_map.min():
            fused_saliency_map = (fused_saliency_map - fused_saliency_map.min())/(fused_saliency_map.max() - fused_saliency_map.min())

        return fused_saliency_map[0,0,:,:], imgClass
