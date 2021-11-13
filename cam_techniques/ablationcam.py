#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 18:26:30 2021

@author: sagar
"""
import torch.nn.functional as F
import torch.nn as nn
import torch

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'


class headModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.maxpool = self.model.features[30]
        self.avgpool = self.model.avgpool
        self.classifier = self.model.classifier
        
    def forward(self, x):
        x = self.maxpool(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x
        

class AblationCAM(nn.Module):
    def __init__(self, model, layer):
        super().__init__()
        self.model = model
        self.layer = layer
        
        def forwardHook(module, input, output):
            self.activations = output
            return None
        
        self.layer.register_forward_hook(forwardHook)
        
        self.headModel = headModel(model).eval()
        
    def forward(self, x, classid=None):
        _, _, ht, wt = x.shape
        with torch.no_grad():
            logits = self.model(x)
            
        activations = self.activations
        
        if classid:
            imgClass = torch.LongTensor([classid])
        else:
            imgClass = torch.argmax(logits)
            
        _, C, u, v = activations.shape
        fused_saliency_map= torch.zeros((1,1,ht,wt)).to(device)
            
        for i in range(C):
            saliency_map = activations[:,i:i+1,:,:]
            saliency_map = F.interpolate(saliency_map, size=(ht,wt), mode='bilinear', align_corners=False)

            activationsNew = activations.clone()
            activationsNew[:,i,:,:] = 0
            with torch.no_grad():
                logitsNew = self.headModel(activationsNew)
            alpha = logits[0,imgClass]-logitsNew[0,imgClass]
            alpha /= logits[0,imgClass]
            fused_saliency_map += alpha*saliency_map
            
        fused_saliency_map = F.relu(fused_saliency_map)
            
        if fused_saliency_map.max() != fused_saliency_map.min():
            fused_saliency_map = (fused_saliency_map - fused_saliency_map.min())/(fused_saliency_map.max() - fused_saliency_map.min())
        
        return fused_saliency_map[0,0,:,:], imgClass
