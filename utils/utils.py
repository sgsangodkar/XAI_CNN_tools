#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 17:48:34 2021

@author: sagar
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2

def show_cam_on_image(img,
                      mask,
                      use_rgb = False,
                      colormap = cv2.COLORMAP_JET):
    """ 
    Overlays the cam mask on the image as an heatmap (default BGR).
    Adapted from https://github.com/jacobgil/pytorch-grad-cam, 
        file: utils/image.py, function: show_cam_on_image.
    
    Arguments:
        img: The base image in RGB or BGR format.
        mask: The cam mask.
        use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
        colormap: The OpenCV colormap to be used.
    
    Output:
        The image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        img = img/np.max(img)
        
    cam = heatmap + img
    cam = cam / np.max(cam)
    
    return np.uint8(255 * cam)


def visualize(img, saliency_map, save_path=None):

    """ 
    Method to save/plot the explanation.
    Adapted from https://github.com/haofanwang/Score-CAM,
        file: utils/__init__.py, function: basic_visualize
        
    Arguments
        img: Tensor or PIL. Original image.
        saliency_map: Tensor. Saliency map result.
        save_path: String. Defaults to None.
           
    """
    use_rgb = True if save_path is not None else False
    
    subplots = [
        ('Input Image', img),
        ('Overlay', show_cam_on_image(img, saliency_map, use_rgb))
    ]

    num_subplots = len(subplots)

    fig = plt.figure()

    for i, (title, img) in enumerate(subplots):
        ax = fig.add_subplot(1, num_subplots, i + 1)
        ax.set_axis_off()
        ax.set_title(title)
        ax.imshow(img)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()