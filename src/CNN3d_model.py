import tensorflow as tf
import numpy as np

FC_SIZE1 = 250
FC_SIZE2 = 250
DTYPE = tf.float32

def _weight_variable(name, shape):
    return tf.get_variable(name, shape, DTYPE, tf.truncated_normal_initializer(stddev=0.1))


def _bias_variable(name, shape):
    return tf.get_variable(name, shape, DTYPE, tf.constant_initializer(0.1, dtype=DTYPE))


def cnn_model(boxes, num_props,num_classes,do_prob):
    prev_layer = boxes
    in_filters = num_props
    
    with tf.variable_scope('conv553a'):
        out_filters = 32
        kernel = _weight_variable('weights', [5, 5, 3, in_filters, out_filters])
        conv = tf.nn.conv3d(prev_layer, kernel, [1, 1, 1, 1, 1], padding='SAME')
        biases = _bias_variable('biases', [out_filters])
        bias = tf.nn.bias_add(conv, biases)
        prev_layer = tf.nn.relu(bias)
        in_filters = out_filters

    with tf.variable_scope('poola'):
        pool = tf.nn.max_pool3d(prev_layer, ksize=[1, 3, 3, 1, 1], strides=[1, 2, 2, 1, 1], padding='SAME')
        prev_layer = pool

    with tf.variable_scope('conv333b'):
        out_filters = 48
        kernel = _weight_variable('weights', [3, 3, 3, in_filters, out_filters])
        conv = tf.nn.conv3d(prev_layer, kernel, [1, 1, 1, 1, 1], padding='SAME')
        biases = _bias_variable('biases', [out_filters])
        bias = tf.nn.bias_add(conv, biases)
        prev_layer = tf.nn.relu(bias)
        in_filters = out_filters

    with tf.variable_scope('poolb'):
        pool = tf.nn.max_pool3d(prev_layer, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
        prev_layer = pool
    
    with tf.variable_scope('conv333c'):
        out_filters = 64
        kernel = _weight_variable('weights', [3, 3, 3, in_filters, out_filters])
        conv = tf.nn.conv3d(prev_layer, kernel, [1, 2, 2, 2, 1], padding='SAME')
        biases = _bias_variable('biases', [out_filters])
        bias = tf.nn.bias_add(conv, biases)
        prev_layer = tf.nn.relu(bias)
        in_filters = out_filters

    with tf.variable_scope('poolc'):
        pool = tf.nn.max_pool3d(prev_layer, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
        prev_layer = pool

    with tf.variable_scope('fc1'):
        dim = np.prod(prev_layer.get_shape().as_list()[1:])
        prev_layer_flat = tf.reshape(prev_layer, [-1, dim])
        weights_1 = _weight_variable('weights', [dim, FC_SIZE1])
        biases_1 = _bias_variable('biases', [FC_SIZE1])
        prev_layer = tf.nn.relu(tf.matmul(prev_layer_flat, weights_1) + biases_1)

    with tf.variable_scope('softmax_linear'):
        dim = np.prod(prev_layer.get_shape().as_list()[1:])
        weights = _weight_variable('weights', [dim, num_classes])
        biases = _bias_variable('biases', [num_classes])
        softmax_linear = tf.add(tf.matmul(prev_layer, weights), biases)

    with tf.variable_scope('regulization'):
        regularizers = tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(biases_1) + tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases)      
    return softmax_linear, regularizers

def loss_function(logits, labels):
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels, DTYPE),logits=logits)
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')

def regulized_loss_function(logits, labels, beta ,reg):
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels, DTYPE),logits=logits)
    return tf.reduce_mean(cross_entropy + beta * reg, name='xentropy_mean')
