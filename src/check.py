#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import mode

test_pred = np.load('test_pred.npy')
test_prob = np.load('test_prob.npy')
ground_truth = np.load('ground_truth.npy')

def softmax(x):
    if x.ndim==1:
        S=np.sum(np.exp(x))
        return np.exp(x)/S
    elif x.ndim==2:
        result=np.zeros_like(x)
        M,N=x.shape
        for n in range(N):
            S=np.sum(np.exp(x[:,n]))
            result[:,n]=np.exp(x[:,n])/S
        return result

def check():
    x = test_prob.transpose()
    k = softmax(x).transpose()
    for label in range(6):
        idx = ground_truth == label
        a = k[idx,:]
        b = test_pred[idx]
        if (a.shape[0]!=0):
            softmax_for_label = np.sum(a,0)
            print('ground_truth:',label)
            print("".join(['%0.02f ' % (softmax_for_label[n]) for n in range(6)]))
            max_idx = np.argmax(softmax_for_label)
            softmax_for_label[max_idx] = 0
            second_idx = np.argmax(softmax_for_label)
            print('maxprob', max_idx, 'secondprob', second_idx,'mode', mode(b)[0][0])

if __name__ == '__main__':          
    check()
