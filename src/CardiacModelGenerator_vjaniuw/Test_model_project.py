#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 16:01:18 2024

@author: vinayjani
"""
import wx

import glob
from pydicom import dcmread
import numpy as np

def vinayDicomSeries():
    app = wx.App(False)
    selector = wx.DirSelector("Choose a folder")
    if not selector :
        return None 
    Images = glob.glob(selector +'/*.dcm')
    
    out_images = np.ndarray((len(Images),),dtype=object)
    for ii in range(len(Images)):
        out_images[ii] = dcmread(Images[ii])
    Num_Slices = np.float64(out_images.shape)
    return  out_images, Num_Slices

[A,B] = vinayDicomSeries()



