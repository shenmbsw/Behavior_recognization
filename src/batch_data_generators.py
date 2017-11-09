import numpy as np
import copy

def concatenate_list(array_list):
    return np.stack(array_list)

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

# to generate frame by frame data
@restartable
def dataset_iterator_fbf(batch_size, data_filelist, label_filelist,file_num):
    assert batch_size > 0 or batch_size == -1
    for i in range(file_num):
        data_filepath = data_filelist[i]
        label_filepath = label_filelist[i]
        data = np.load(data_filepath)
        data = np.expand_dims(data, axis=3)
        data = np.expand_dims(data, axis=4)
        label = np.load(label_filepath).reshape(-1,1)
        label = ((np.arange(7) == label[:, None]).astype(int))
        label = label.squeeze()
        size = label.shape[0]
        curind = 0
        if batch_size==-1:
            batch_size = size
        for j in range(int(size/batch_size)):
            if batch_size==-1:
                yield data,label
            else:
                if curind+batch_size<=size:
                    X = data[curind:curind+batch_size,:,:,:]
                    Y = label[curind:curind+batch_size,:]
                    curind = curind + batch_size
                    yield X,Y

                    
    
