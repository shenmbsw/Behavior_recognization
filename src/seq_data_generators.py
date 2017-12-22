import numpy as np
import copy
from skimage.transform import resize,rotate
import random

def data_aug(input_frame,resize_rate=0.9, angel=0):
    gate = random.random()
    flip = random.randint(0, 1)
    size = input_frame.shape[0]
    deep = input_frame.shape[2]
    rsize = random.randint(np.floor(resize_rate*size),size)
    output_frame = np.zeros([size,size,deep],dtype = 'uint8')
    if gate < 0.25:
        return input_frame
    else:
        w_s = random.randint(0,size - rsize)
        h_s = random.randint(0,size - rsize)
        rotate_angel = random.randint(-angel,angel)
        for i in range(deep):
            window = rotate(input_frame,rotate_angel)
            window = window[w_s:w_s+size,h_s:h_s+size,i]
            if flip:
                window = window[:,::-1]
            output_frame[:,:,i] = np.floor(resize(window,(size,size),mode='reflect') * 255)
        return output_frame


class GeneratorRestartHandler(object):
    def __init__(self, gen_func, argv, kwargv):
        self.gen_func = gen_func
        self.argv = copy.copy(argv)
        self.kwargv = copy.copy(kwargv)
        self.local_copy = self.gen_func(*self.argv, **self.kwargv)

    def __iter__(self):
        return GeneratorRestartHandler(self.gen_func, self.argv, self.kwargv)

    def __next__(self):
        return next(self.local_copy)

    def next(self):
        return self.__next__()


def restartable(g_func):
    def tmp(*argv, **kwargv):
        return GeneratorRestartHandler(g_func, argv, kwargv)
    return tmp


@restartable
def dataset_iterator_seg_batch_3D(batch_size, work_dir, data_filelist, frame_depth, stride):
    file_num = len(data_filelist)
    batch_idx = 0
    for i in range(file_num):
        data_filepath = work_dir + data_filelist[i]
        data = np.load(data_filepath)
        data = np.expand_dims(data,4)
        data = np.expand_dims(data,0)
        size = data.shape[3]
        label = np.zeros((1,6))
        label[:,int(data_filelist[i][-5])] = 1
        curind = 0
        if batch_size == 1:
            while (curind < size-frame_depth):
                X = data[:,:,:,curind:curind+frame_depth,:]
                Y = label
                curind = curind + stride
                yield X,Y
        else:
            while (curind < size-frame_depth):
                if batch_idx == 0:
                    X = data[:,:,:,curind:curind+frame_depth,:]
                    Y = label
                    curind = curind + stride
                    batch_idx = batch_idx+1
                elif batch_idx < batch_size-1:
                    X = np.concatenate([X,data[:,:,:,curind:curind+frame_depth,:]],axis=0)
                    Y = np.concatenate([Y,label],axis=0)
                    curind = curind + stride
                    batch_idx = batch_idx+1
                elif batch_idx == batch_size-1:
                    X = np.concatenate([X,data[:,:,:,curind:curind+frame_depth,:]])
                    Y = np.concatenate([Y,label],axis=0)
                    curind = curind + stride
                    batch_idx = 0
                    yield X,Y
                  
@restartable
def dataset_iterator_aug_batch_3D(batch_size, work_dir, data_filelist, frame_depth, stride):
    file_num = len(data_filelist)
    batch_idx = 0
    for i in range(file_num):
        data_filepath = work_dir + data_filelist[i]
        data = np.load(data_filepath)
        data = data_aug(data,0.9,2)
        data = np.expand_dims(data,4)
        data = np.expand_dims(data,0)
        size = data.shape[3]
        label = np.zeros((1,6))
        label[:,int(data_filelist[i][-5])] = 1
        curind = 0
        if batch_size == 1:
            while (curind < size-frame_depth):
                X = data[:,:,:,curind:curind+frame_depth,:]
                Y = label
                curind = curind + stride
                yield X,Y
        else:
            while (curind < size-frame_depth):
                if batch_idx == 0:
                    X = data[:,:,:,curind:curind+frame_depth,:]
                    Y = label
                    curind = curind + stride
                    batch_idx = batch_idx+1
                elif batch_idx < batch_size-1:
                    X = np.concatenate([X,data[:,:,:,curind:curind+frame_depth,:]],axis=0)
                    Y = np.concatenate([Y,label],axis=0)
                    curind = curind + stride
                    batch_idx = batch_idx+1
                elif batch_idx == batch_size-1:
                    X = np.concatenate([X,data[:,:,:,curind:curind+frame_depth,:]])
                    Y = np.concatenate([Y,label],axis=0)
                    curind = curind + stride
                    batch_idx = 0
                    yield X,Y 


@restartable            
def dataset_iterator_interval_batch(batch_size, work_dir, data_filelist):
    file_num = len(data_filelist)
    batch_idx = 0  
    for i in range(file_num):
        data_filepath = work_dir + data_filelist[i]
        data = np.load(data_filepath)
        data = np.expand_dims(data,0)
        size = data.shape[3]
        label = np.zeros((1,6))
        label[:,int(data_filelist[i][-5])] = 1
        curind = 0
        for j in range(size-16):
            if batch_idx == 0:
                X1 = data[:,:,:,curind:curind+1]
                X2 = data[:,:,:,curind+4:curind+5]
                X3 = data[:,:,:,curind+8:curind+9]
                X4 = data[:,:,:,curind+12:curind+13]
                Y = label
                curind = curind + 1
                batch_idx = batch_idx+1
            elif batch_idx < batch_size-1:
                X1 = np.concatenate([X1,data[:,:,:,curind:curind+1]],axis=0)
                X2 = np.concatenate([X2,data[:,:,:,curind+4:curind+5]],axis=0)
                X3 = np.concatenate([X3,data[:,:,:,curind+8:curind+9]],axis=0)
                X4 = np.concatenate([X4,data[:,:,:,curind+12:curind+13]],axis=0)
                Y = np.concatenate([Y,label],axis=0)
                curind = curind + 1
                batch_idx = batch_idx+1
            elif batch_idx == batch_size-1:
                X1 = np.concatenate([X1,data[:,:,:,curind:curind+1]],axis=0)
                X2 = np.concatenate([X2,data[:,:,:,curind+4:curind+5]],axis=0)
                X3 = np.concatenate([X3,data[:,:,:,curind+8:curind+9]],axis=0)
                X4 = np.concatenate([X4,data[:,:,:,curind+12:curind+13]],axis=0)
                Y = np.concatenate([Y,label],axis=0)
                curind = curind + 1
                batch_idx = 0
                X = np.concatenate([X1,X2,X3,X4],axis= 3)
                yield X,Y

@restartable
def dataset_iterator_seg_batch(batch_size, work_dir, data_filelist, frame_depth):
    file_num = len(data_filelist)
    batch_idx = 0    
    for i in range(file_num):
        data_filepath = work_dir + data_filelist[i]
        data = np.load(data_filepath)
        data = np.expand_dims(data,0)
        size = data.shape[3]
        label = np.zeros((1,6))
        label[:,int(data_filelist[i][-5])] = 1
        curind = 0
        for j in range(size-frame_depth):
            if batch_idx == 0:
                X = data[:,:,:,curind:curind+frame_depth]
                Y = label
                curind = curind + 1
                batch_idx = batch_idx+1
            elif batch_idx < batch_size-1:
                X = np.concatenate([X,data[:,:,:,curind:curind+frame_depth]],axis=0)
                Y = np.concatenate([Y,label],axis=0)
                curind = curind + 1
                batch_idx = batch_idx+1
            elif batch_idx == batch_size-1:
                X = np.concatenate([X,data[:,:,:,curind:curind+frame_depth]])
                Y = np.concatenate([Y,label],axis=0)
                curind = curind + 1
                batch_idx = 0
                yield X,Y            


@restartable            
def dataset_iterator_singleframe_batch(batch_size,work_dir, data_filelist):
    file_num = len(data_filelist)
    batch_idx = 0
    for i in range(file_num):
        data_filepath = work_dir + data_filelist[i]
        data = np.load(data_filepath)
        data = np.expand_dims(data,0)
        size = data.shape[3]
        label = np.zeros((1,6))
        label[:,int(data_filelist[i][-5])] = 1
        curind = 0
        for j in range(size):
            if batch_idx == 0:
                X = data[:,:,:,curind:curind+1]
                Y = label
                curind = curind + 1
                batch_idx = batch_idx+1
            elif batch_idx < batch_size-1:
                X = np.concatenate([X,data[:,:,:,curind:curind+1]],axis=0)
                Y = np.concatenate([Y,label],axis=0)
                curind = curind + 1
                batch_idx = batch_idx+1
            elif batch_idx == batch_size-1:
                X = np.concatenate([X,data[:,:,:,curind:curind+1]])
                Y = np.concatenate([Y,label],axis=0)
                curind = curind + 1
                batch_idx = 0
                yield X,Y
