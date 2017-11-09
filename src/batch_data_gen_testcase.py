import batch_data_generators as bdg

train_data_filelist = []
train_label_filelist = []

for i in range(9):
    train_data_filelist.append('array/X_seq_%d.npy'%(i+1))
    train_label_filelist.append('array/Y_seq_%d.npy'%(i+1))
print(train_data_filelist)
print(train_label_filelist)

test_data_filelist = []
test_label_filelist = []

for i in range(1):
    test_data_filelist.append('array/X_seq_%d.npy'%(10-i))
    test_label_filelist.append('array/Y_seq_%d.npy'%(10-i))

dataset_generators = {
        'train': bdg.dataset_iterator_fbf(16,train_data_filelist, train_label_filelist,9),
        'test':  bdg.dataset_iterator_fbf(16,test_data_filelist, test_label_filelist,1),
    }

for iter_i, data_batch in enumerate(dataset_generators['train']):
    if(data_batch[0].shape[0] != data_batch[1].shape[0]):
        print('kkk',iter_i)
        print(iter_i*16)
        print(data_batch[0].shape[0])
        print(data_batch[1].shape[0])
    if(iter_i%100 == 0):
        print(iter_i)
        
for iter_i, data_batch in enumerate(dataset_generators['test']):
    if(data_batch[0].shape[0] != data_batch[1].shape[0]):
        print('kkk',iter_i)
        print(iter_i*16)
        print(data_batch[0].shape[0])
        print(data_batch[1].shape[0])
    if(iter_i%100 == 0):
        print(iter_i)