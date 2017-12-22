import tensorflow as tf
import os
#from CNN3d_model import cnn_model, loss_function, regulized_loss_function
from CNN2d_model import cnn_2d_model, loss_function
from check import check
import numpy as np
import seq_data_generators as sdg

# Get the assiged number of cores for this job. This is stored in
# the NSLOTS variable, If NSLOTS is not defined throw an exception.
def get_n_cores():
  nslots = os.getenv('NSLOTS')
  if nslots is not None:
    return int(nslots)
  raise ValueError('Environment variable NSLOTS is not defined.')  
  
def apply_classification_loss(model_function,dev):
    with tf.Graph().as_default() as g:
        with tf.device(dev):
            frame_depth = 8
            num_classes = 6
            with tf.name_scope('input'):
                x_ = tf.placeholder(tf.float32, [None, 96, 96, frame_depth, 1],name='x')
                y_ = tf.placeholder(tf.int32, [None,num_classes],name='y')
            softmax_linear,regularizers = model_function(x_,1,num_classes)
            with tf.name_scope('loss'):
                cross_entropy_loss = loss_function(softmax_linear,y_)
#                cross_entropy_loss = regulized_loss_function(softmax_linear,y_,0.01,regularizers)
            with tf.name_scope('train'):
                trainer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
                train_op = trainer.minimize(cross_entropy_loss)
            with tf.name_scope('result'):
                prediction = tf.argmax(softmax_linear,1)
                y_truth = tf.argmax(y_,1)
            with tf.name_scope("accuracy"):
                _,accuracy = tf.metrics.accuracy(y_truth, prediction, name='accuracy')
            tf.summary.scalar('accuracy', accuracy)
            tf.summary.scalar('loss', cross_entropy_loss)
            merged = tf.summary.merge_all()
        model_dict = {'graph': g, 'inputs': [x_, y_], 'train_op': train_op,
                      'summary':merged, 'predict_result': prediction,
                      'pred_prob':softmax_linear,'loss':cross_entropy_loss}
    return model_dict

def train_model(model_dict, train_dataset, epoch_n, broad_dir, restore = False):
    session_conf = tf.ConfigProto(
          intra_op_parallelism_threads=get_n_cores()-1,
          inter_op_parallelism_threads=1,
          allow_soft_placement=True, 
          log_device_placement=True)
    max_iter = 1800
    with model_dict['graph'].as_default():
        saver = tf.train.Saver(max_to_keep=0) 
        with tf.Session(config=session_conf) as sess:
            if restore:
                saver.restore(sess, "./Model/3dconv_final.ckpt") 
            else:
                sess.run(tf.variables_initializer(tf.global_variables()))
                
            sess.run(tf.variables_initializer(tf.local_variables()))
            acc_count = tf.local_variables()[-1]
            acc_total = tf.local_variables()[-2]
            train_writer = tf.summary.FileWriter(broad_dir+'/cnn3d', model_dict['graph'])
            for epoch_i in range(epoch_n):
                for iter_i, data_batch in enumerate(train_dataset):
                    train_feed_dict = dict(zip(model_dict['inputs'], data_batch))
              
                    _, summary = sess.run([model_dict['train_op'],model_dict['summary']],
                                          feed_dict=train_feed_dict)
                    train_writer.add_summary(summary, epoch_i * max_iter + iter_i)
                sess.run([acc_count.assign(0),acc_total.assign(0)])
                if(epoch_i == 2):
                    saver.save(sess, "Model/3dconv_train%d.ckpt"%epoch_i)
                if(epoch_i %5 == 0):
                    saver.save(sess, "Model/3dconv_train%d.ckpt"%epoch_i)
                saver.save(sess, "Model/3dconv_final.ckpt")

def test_model(model_dict, test_dataset, epoch):
    flag = 0
    gt = []
    with model_dict['graph'].as_default():
        saver = tf.train.Saver() 
        with tf.Session() as sess:
            if (epoch == -1):
                saver.restore(sess, "./Model/3dconv_final.ckpt")
            else:
                saver.restore(sess, "./Model/3dconv_train%d.ckpt"%epoch) 
            for iter_i, data_batch in enumerate(test_dataset):
                test_feed_dict = dict(zip(model_dict['inputs'], data_batch))
                sess_list = [model_dict['predict_result'],model_dict['pred_prob'],model_dict['loss']]
                y_pred,y_prob,loss = sess.run(sess_list, test_feed_dict)
                if flag == 0:
                    test_pred = y_pred
                    test_prob = y_prob
                    flag = 1
                else:
                    test_pred = np.concatenate([test_pred,y_pred])
                    test_prob = np.concatenate([test_prob,y_prob])
                print(loss)
                y_truth = np.argmax(data_batch[1])
                gt.append(y_truth)
    np.save('test_pred.npy',test_pred)
    np.save('test_prob.npy',test_prob)
    np.save('ground_truth.npy',np.array(gt))    

def write():
    model_dict = apply_classification_loss(cnn_2d_model,"/cpu:0")
    tf.summary.FileWriter('graph/', model_dict['graph'])

def train():
    array_dir = "array_seg/"
    train_list = os.listdir(array_dir+'train')
    train_dataset = sdg.dataset_iterator_seg_batch_3D(1,array_dir+'train/', train_list,8,4)
    model_dict = apply_classification_loss(cnn_model,"/gpu:0")
    train_model(model_dict, train_dataset, 30, 'graph/',restore = False)

def test(i):
    array_dir = "array_seg/"
    test_list = os.listdir(array_dir+'test')
    test_dataset = sdg.dataset_iterator_seg_batch_3D(1,array_dir+'test/', test_list,8,4)
    model_dict = apply_classification_loss(cnn_model,"/cpu:0")
    test_model(model_dict, test_dataset,i)

if __name__ == "__main__":
    write()
#    train()
#    cp_list = [0,2,5,10,15,25]
#    for i in cp_list:
#        test(i)
#        check()
