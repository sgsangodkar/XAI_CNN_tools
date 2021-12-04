#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 17:21:46 2021

@author: sagar
"""
import torch
from torch.autograd import grad

a = torch.Tensor((1,2,3,4))
a.requires_grad = True

y = (a + a**2 + a**3)

#y.backward()

grads = grad(outputs=y.unbind(), inputs=a)


a = torch.Tensor(((1,2,3,4),(1,2,4,5)))
a.requires_grad = True

y = (a + a**2)

#y.backward()

grads = grad(outputs=y.unbind()[:][0].unbind(), inputs=a)