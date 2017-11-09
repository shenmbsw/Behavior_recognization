import numpy as np
try:
    from itertools import izip as zip
except ImportError:
    print('This is Python 3')

def padding(input,ref_shape):
    for i in range(len(input)):
        temp = np.zeros(ref_shape)
        insertHere = [slice(0,input[i].shape[dim]) for dim in range(input[i].ndim)]
        # Insert the array in the result at the specified offsets
        temp[insertHere] = input[i]
        input[i] = temp
    return input

def fast_pad(input, ref_shape):
    assert len(ref_shape)==3
    for i in range(len(input)):
        input[i] = np.lib.pad(input[i],((0,ref_shape[0]-input[i].shape[0]),(0,ref_shape[1]-input[i].shape[1]),(0,ref_shape[2]-input[i].shape[2])),'constant',constant_values = 0)
    return input

def concatenate_list(array_list):
    return np.stack(array_list)

def dataset_3d_iterator(batch_size, data_filepath, label_filepath):
    assert batch_size > 0 or batch_size == -1
    data = np.load(data_filepath)
    label = np.load(label_filepath).reshape(19,1)
    size = np.prod(label.shape)
    curind = 0
    while True:
        if batch_size==-1:
            yield data,label
        else:
            if curind+batch_size<=size:
                temp = curind
                curind = curind+batch_size
                result = concatenate_list(fast_pad(data[temp:curind],[470,512,512]))
                yield result,label[temp:curind]
            else:
                temp = curind
                curind = batch_size-(size-curind)
                ind = [i for i in range(temp,size)] + [i for i in range(0,curind)]
                result = concatenate_list(fast_pad(data[ind],[470,512,512]))
                yield result, label[ind]

def main():
    data_filepath = 'temp.npy'
    label_filepath = 'label.npy'
    batch_size = 5
    dataset_generators = {
            'train': dataset_3d_iterator(batch_size, data_filepath, label_filepath),
            'test':  dataset_3d_iterator(batch_size, data_filepath, label_filepath),
        }
    model_dict = apply_classification_loss_rnn(cnn_rnn, batch_size)
    train_model(model_dict, batch_size, dataset_generators, 100, 1)


main()
