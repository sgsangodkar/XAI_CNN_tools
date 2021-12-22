# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch.autograd import grad

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

class IG(object):
    def __init__(self, model, layer, steps=32):
        self.model = model
        self.layer = layer
        self.steps = steps
        
        
        
    def __call__(self, x, classid=None):
        with torch.no_grad():
            logits = self.model(x)
        
        if classid:
            imgClass = torch.LongTensor([classid])
        else:
            imgClass = torch.argmax(logits)
                  
            
        
        #alpha = torch.arange(0,self.steps).to(device) + 1
        alpha = torch.linspace(0,1,self.steps+1)[1:].to(device)
        x_b = torch.zeros(x.shape).to(device)
        images = alpha[:,None,None,None]*(x-x_b)
        images += x_b.expand_as(images)
        images.requires_grad_()
        
        logits = self.model(images)
        grads = grad(outputs=logits[:,imgClass].unbind(), inputs=images)
        
        output = grads[0].mean(dim=0, keepdim=True)
        
        output = (x-x_b)*output[0]
        
        output = output.mean(dim=0)
        

        
        return F.relu(output)
        
        
        
