import os
import cv2
import numpy as np

cr_path = "F:/Code/Datasets/NTIRE2026/color_track/Train/IN_CR_COM/"
sh_path = "F:/Code/Datasets/NTIRE2026/color_track/Train/IN_SH_COM/"
def read_cr(cr_path):
     for file in os.listdir(cr_path):
         img = cv2.imread(os.path.join(cr_path, file), cv2.IMREAD_GRAYSCALE)
         print('cr_path: ', file, 'img.shape: ', img.shape)

def read_sh(sh_path):
    for file in os.listdir(sh_path):
        img = cv2.imread(os.path.join(sh_path, file), cv2.IMREAD_GRAYSCALE)
        print('sh_path: ', file, 'img.shape: ', img.shape)

#read_cr(cr_path)
read_sh(sh_path)
