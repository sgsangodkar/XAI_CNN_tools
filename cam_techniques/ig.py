# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

class IG(object):
    def __init__(self, model, layer, steps=30, batch_size=16):
        self.model = model
        self.layer = layer
        self.steps = steps
        self.batch_size = batch_size
        
        
        
    def __call__(self, x, classid=None):
        with torch.no_grad():
            logits = self.model(x)
        
        if classid:
            imgClass = torch.LongTensor([classid])
        else:
            imgClass = torch.argmax(logits)
                  
        
        x_b = torch.zeros(x.shape).to(device)
       
        
        alpha = torch.arange(0,self.steps).to(device) + 1
        
        images = alpha[:,None,None,None]*x
        images += x_b.expand_as(images)
        
        gradients = []
        for image in images:
            inp = image.unsqueeze(0)
            inp.requires_grad = True
            logits = self.model(inp)
            self.model.zero_grad()
            logits[0,imgClass].backward()
            gradients.append(inp.grad.data)
            
        #gradients/=self.steps
        #gradients*=(x-x_b)
        
        return gradients
        
        
        