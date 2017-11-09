import tensorflow as tf
import numpy as np
import batch_data_generators as bdg
from ftf_model import cnn_model, loss_function

 
def apply_classification_loss_rnn(model_function):
    with tf.Graph().as_default() as g:
        frame_depth = 1
        num_labels = 7
        with tf.device("/cpu:0"):
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

    with model_dict['graph'].as_default(), tf.Session() as sess:
        sess.run(tf.variables_initializer(tf.global_variables()))
        train_record = ()
        for epoch_i in range(epoch_n):
            train_collect = []
            for iter_i, data_batch in enumerate(dataset_generators['train']):
                train_feed_dict = dict(zip(model_dict['inputs'], data_batch))
                train_to_compute = [model_dict['train_op'],model_dict['loss'],model_dict['accuracy']]
                sess.run(train_to_compute, feed_dict=train_feed_dict)
                train_collect.append(train_to_compute[1:])
            train_averages = np.mean(train_collect,axis=0)
            train_record += tuple(train_averages)
        print(train_record)
     
           
data_filelist = []
label_filelist = []
for i in range(10):
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

