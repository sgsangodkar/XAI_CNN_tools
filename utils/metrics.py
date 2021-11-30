#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation metrics
"""

import os
import torch
import torch.nn.functional as f

class MetricsMeter():
    def __init__(self):
        self.drop_in_conf = 0
        self.drop_in_act = 0
        self.inc_in_conf = 0
        self.inc_in_act = 0
        
    def update(self, logits, logits_new, cid):
        prob = f.softmax(logits, dim=0)
        prob_new = f.softmax(logits_new, dim=0)
        self.drop_in_conf += ((f.relu(prob[cid]-prob_new[cid]))/prob[cid]).item()
        self.drop_in_act += ((f.relu(logits[cid]-logits_new[cid]))/logits[cid]).item()
        self.inc_in_conf += (prob_new[cid] > prob[cid]).item()
        self.inc_in_act += (logits_new[cid] > logits[cid]).item()
    
    def average(self, N):
        self.drop_in_conf /= N
        self.drop_in_act /= N
        self.inc_in_conf /= N
        self.inc_in_act /= N      
 