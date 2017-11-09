import tensorflow as tf
import numpy as np
import os
import batch_data_generators as bdg
from ftf_model import cnn_model, loss_function

# Get the assiged number of cores for this job. This is stored in
# the NSLOTS variable, If NSLOTS is not defined throw an exception.
def get_n_cores():
  nslots = os.getenv('NSLOTS')
  if nslots is not None:
    return int(nslots)
  raise ValueError('Environment variable NSLOTS is not defined.')

 
def apply_classification_loss_rnn(model_function):
    with tf.Graph().as_default() as g:
        frame_depth = 1
        num_labels = 7
        with tf.device("/gpu:0"):
            trainer = tf.train.AdamOptimizer()
            x_ = tf.placeholder(tf.float32, [None,480, 720, frame_depth,1])
            y_ = tf.placeholder(tf.int32, [None,num_labels])
            y_logits = model_function(x_,1,num_labels)
            loss = loss_function(y_logits,y_)
            cross_entropy_loss = tf.reduce_mean(loss)
            train_op = trainer.minimize(cross_entropy_loss)
            y_prob = tf.sigmoid(y_logits)
            y_pred = tf.cast(tf.round(y_prob), tf.int32)
            correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    model_dict = {'graph': g, 'inputs': [x_, y_], 'train_op': train_op,
                  'pred':y_pred,'prob':y_prob,
                  'accuracy': accuracy, 'loss': cross_entropy_loss}
    return model_dict

def train_model(model_dict, dataset_generators, epoch_n):
    session_conf = tf.ConfigProto(
          intra_op_parallelism_threads=get_n_cores()-1,
          inter_op_parallelism_threads=1,
          allow_soft_placement=True, 
          log_device_placement=True)

    with model_dict['graph'].as_default():
        sess = tf.Session(config=session_conf)
        sess.run(tf.variables_initializer(tf.global_variables()))
        train_collect = []
        for epoch_i in range(epoch_n):
            for iter_i, data_batch in enumerate(dataset_generators['train']):
                train_feed_dict = dict(zip(model_dict['inputs'], data_batch))
                train_to_compute = [model_dict['train_op'],model_dict['loss'],model_dict['accuracy']]
                sess.run(train_to_compute, feed_dict=train_feed_dict)
                train_collect.append(train_to_compute[1:])
                if (iter_i%100==1):
                    print(train_to_compute[1:])
            train_average = np.mean(train_collect,axis=0)
            print(train_average)
           
data_filelist = []
label_filelist = []
for i in range(3):
    data_filelist.append('array/X_seq_%d.npy'%(i+1))
    label_filelist.append('array/Y_seq_%d.npy'%(i+1))
print(data_filelist)
print(label_filelist)

dataset_generators = {
        'train': bdg.dataset_iterator_fbf(16,data_filelist, label_filelist,10),
        'test':  bdg.dataset_iterator_fbf(16,data_filelist, label_filelist,10),
    }

model_dict = apply_classification_loss_rnn(cnn_model)
train_model(model_dict, dataset_generators, 2)

