#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 22:36:01 2017

@author: shen
"""

import numpy as np
import cv2
#from sklearn.cluster import KMeans

cap = cv2.VideoCapture('seq8.avi')
fgbg = cv2.createBackgroundSubtractorMOG2()
idx = 0
center = np.array([[0,0],[0,0]])

while(cap.isOpened()):
    idx +=1
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)

#    if (idx%20 == 0):
#        loc = np.argwhere(fgmask>30)
#        kmeans = KMeans(n_clusters=2, random_state=0).fit(loc)
#        center = kmeans.cluster_centers_
#        
#    motion_center_x = (center[0,1] + center[1,1])/2
#    motion_center_y = (center[0,0] + center[1,0])/2
#    
#    width = 1.8*abs(center[0,1] - center[1,1])
#    height = width/720*480
#    
#    pt1 = (int(motion_center_x-0.5*width), int(motion_center_y-0.5*height))
#    pt2 = (int(motion_center_x+0.5*width), int(motion_center_y+0.5*height))
    
#    pt1 = (220,113)
#    pt2 = (620,410)
    pt1 = (100,160)
    pt2 = (445,400)


    cv2.circle(fgmask,(int(center[0,1]),int(center[0,0])),10,(255,255,255))
    cv2.circle(fgmask,(int(center[1,1]),int(center[1,0])),10,(255,255,255))
    cv2.rectangle(fgmask,pt1,pt2,(255,255,255))
    
    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(20) & 0xff
    if k == 27:
        break
    elif k == 99:
        cv2.imwrite('fgmask3.jpg',fgmask)
cap.release()
cv2.destroyAllWindows()
