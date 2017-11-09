import numpy as np
import batch_data_generators as bdg

data_filelist = []
label_filelist = []
for i in range(10):
    data_filelist.append('array/X_seq_%d.npy'%(i+1))
    label_filelist.append('array/Y_seq_%d.npy'%(i+1))
print(data_filelist)
print(label_filelist)

#for i in range(10):
#    d = np.load(data_filelist[i])
#    l = np.load(label_filelist[i])
#    print(d.shape)
#    print(l.shape)

dataset_generators = {
        'train': bdg.dataset_iterator_fbf(16,data_filelist, label_filelist,10),
        'test':  bdg.dataset_iterator_fbf(16,data_filelist, label_filelist,10),
    }

for iter_i, data_batch in enumerate(dataset_generators['train']):
    if(data_batch[0].shape[0] != data_batch[1].shape[0]):
        print('kkk',iter_i)
        print(iter_i*16)
        print(data_batch[0].shape[0])
        print(data_batch[1].shape[0])