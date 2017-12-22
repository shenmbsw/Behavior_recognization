from scipy.misc import imread, imsave
import numpy as np
import os
from skimage import transform
import random

def data_aug(input_frame,angel=5,resize_rate=0.9):
    flip = random.randint(0, 1)
    size = input_frame.shape[0]
    rsize = random.randint(np.floor(resize_rate*size),size)
    output_frame = np.zeros([size,size],dtype = 'uint8')
    w_s = random.randint(0,size - rsize)
    h_s = random.randint(0,size - rsize)
    sh = random.random()/2-0.25
    rotate_angel = random.randint(-angel,angel)
    # Create Afine transform
    afine_tf = transform.AffineTransform(shear=sh)
    # Apply transform to image data
    window = transform.warp(input_frame, inverse_map=afine_tf)
    window = transform.rotate(window,rotate_angel)
    window = window[w_s:w_s+size,h_s:h_s+size]
    if flip:
        window = window[:,::-1]
    output_frame = np.floor(transform.resize(window,(size,size),mode='reflect') * 255)
    return output_frame

pos_sam_dir = 'train/'
aug_sam_dir = 'aug/'
#0:607,1:739,2:306,3:792,4:364,5:482,n:1962
k = [40,35,80,30,70,50,15]
split_num = []
for i in range(7):
    label_dir = pos_sam_dir+'%d/'%i
    aug_dir = aug_sam_dir+'%d/'%i
    print(aug_dir)
    img_list = os.listdir(label_dir)
    for img_name in img_list:
        if img_name[-4:]=='.jpg':
            for j in range(k[i]):
                aug_path = aug_dir+'aug%d-%s'%(j,img_name)
                this_img = imread(label_dir+img_name)
                if random.random()<0.9:
                    aug_img = data_aug(this_img)
                else:
                    aug_img = this_img
                print(aug_path)
                imsave(aug_path, aug_img, format=None)
