from scipy.misc import imread
import numpy as np
import ffmpy
import os

W = 720
H = 480

work_dir = '/home/shen/BU_learn/EC720/Final_project'

for s in range(1,3):
    video_dir = work_dir + '/set_1/seq%d.avi'%s
    ff = ffmpy.FFmpeg(
            inputs={video_dir: None},
            outputs={'output%d.png': '-vf fps=15'}
            )
    print(ff)
    ff.run()
    
    file_list = os.listdir()
    match = [x for x in file_list if x[-3:]=='png']
    img_num = len(match)
    
    video_cotainer = np.zeros([H,W,img_num],dtype = 'uint8')
    
    j = 0
    for i in match:
        frame_arr = imread(i,'grey')
    #    print(np.shape(frame_arr))
        video_cotainer[:,:,j] = frame_arr
        j += 1;
        os.remove(i)
    
    #print(video_cotainer[1:5,1:5,3])
    np.save('array/seq_%d.npy'%s,video_cotainer)