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

        

class XGradCAM(nn.Module):
    def __init__(self, model, layer):
        super().__init__()
        self.model = model
        self.layer = layer
        
        def forwardHook(module, input, output):
            self.activations = output
            return None
        
        def backwardHook(module, gradinput, gradoutput):
            self.gradients = gradoutput
            return None
        
        self.layer.register_forward_hook(forwardHook)
        self.layer.register_backward_hook(backwardHook)
        
        
    def forward(self, x, classid=None):
        _, _, ht, wt = x.shape
        
        logits = self.model(x).squeeze()
            
        activations = self.activations
        
        if classid:
            imgClass = torch.LongTensor([classid])
        else:
            imgClass = torch.argmax(logits)
            
        logits[imgClass].backward()
        
        gradients = self.gradients[0]
        
        _, C, u, v = activations.shape
        fused_saliency_map= torch.zeros((1,1,u,v)).to(device)
            
        for i in range(C):
            activation_map = activations[:,i:i+1,:,:]
            gradient_map = gradients[:,i:i+1,:,:]
            alpha = torch.sum(activation_map*gradient_map)
            if torch.sum(activation_map):
                alpha /= torch.sum(activation_map)
            

            fused_saliency_map += alpha*activation_map
            
        fused_saliency_map = F.interpolate(fused_saliency_map, size=(ht,wt), mode='bilinear', align_corners=False)
        fused_saliency_map = F.relu(fused_saliency_map)

        if fused_saliency_map.max() != fused_saliency_map.min():
            fused_saliency_map = (fused_saliency_map - fused_saliency_map.min())/(fused_saliency_map.max() - fused_saliency_map.min())
        
        return fused_saliency_map[0,0,:,:].detach(), logits
