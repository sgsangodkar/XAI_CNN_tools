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
    def __init__(self, model, layer, batch_size=32):
        self.model = model
        self.layer = layer
        self.activations = {}
        self.batch_size = batch_size

        def forward_hook(module, input, output):
            self.activations['activations'] = output.to(device)
            return None
        
        self.layer.register_forward_hook(forward_hook)
        
    def __call__(self, input, classid=None):
        with torch.no_grad():
            logits = self.model(input).squeeze()
        _, _, ht, wt = input.shape
        
        if classid:
            imgClass = torch.LongTensor([classid])
        else:
            imgClass = torch.argmax(logits)
            

        activations = self.activations['activations']
        activations = F.interpolate(activations, size=(ht, wt), mode='bilinear', align_corners=False)

        fused_saliency_map= torch.zeros((ht,wt)).to(device)
        
        maxs = activations.view(activations.shape[0], activations.shape[1], -1).max(dim=-1)[0]
        mins = activations.view(activations.shape[0], activations.shape[1], -1).min(dim=-1)[0]
       
        mask = maxs!=mins
        maxs, mins = maxs[:, :, None, None], mins[:, :, None, None]
        activations[mask,:] = (activations[mask,:] - mins[mask,:]) / (maxs[mask] - mins[mask])
        new_input = input[:,None,:,:,:]*activations[:,:,None,:,:]
        
        weights = torch.empty(new_input.shape[1]).to(device)
        
        for i in range(0, new_input.shape[1], self.batch_size):
            batch = new_input[0,i:i+self.batch_size,:,:,:]
            with torch.no_grad():
                ops = F.softmax(self.model(batch), dim=-1)
            weights[i:i+self.batch_size] = ops[:, imgClass]
        
        
        fused_saliency_map = torch.sum(activations[0]*weights[:,None,None], dim=0)  

   
                
        fused_saliency_map = F.relu(fused_saliency_map)

        if fused_saliency_map.max() != fused_saliency_map.min():
            fused_saliency_map = (fused_saliency_map - fused_saliency_map.min())/(fused_saliency_map.max() - fused_saliency_map.min())

        return fused_saliency_map[:,:].detach(), logits
