# ------------------------------------------------------------------
# Tensorflow implementation of
#  "Learning Dynamic Memory Networks for Object Tracking", ECCV,2018
# Licensed under The MIT License [see LICENSE for details]
# Written by Tianyu Yang (tianyu-yang.com)
# ------------------------------------------------------------------

import tensorflow as tf
import config

def extract_feature(is_train, img_patch):

    conv1 = conv2d_bn_relu(is_train, img_patch, 96, [11, 11], [2, 2], 'valid', name='conv1')
    pool1 = tf.layers.max_pooling2d(conv1, [3, 3], [2, 2], 'valid', name='pool1')
    conv2 = conv2d_bn_relu(is_train, pool1, 256, [5, 5], [1, 1], 'valid', name='conv2')
    pool2 = tf.layers.max_pooling2d(conv2, [3, 3], [2, 2], 'valid', name='pool2')
    conv3 = conv2d_bn_relu(is_train, pool2, 384, [3, 3], [1, 1], 'valid', name='conv3')
    conv4 = conv2d_bn_relu(is_train, conv3, 384, [3, 3], [1, 1], 'valid', name='conv4')
    conv5 = tf.layers.conv2d(conv4, 256, [3, 3], [1, 1], 'valid', name='conv5')

    return conv5

def conv2d(input, filters, kernel_size, strides, padding, name, group=1):

    if group == 1:
        conv = tf.layers.conv2d(input, filters, kernel_size, strides, padding, name=name)
    else:
        input_group = tf.split(input, group, 3)
        conv_group = [tf.layers.conv2d(input, filters//group, kernel_size, strides, padding, name=name+'group_{}'.format(i))
                      for i, input in enumerate(input_group)]
        conv = tf.concat(conv_group, 3)
    return conv

def conv2d_bn_relu(is_train, input, filters, kernel_size, strides, padding, name, group=1):

    conv = conv2d(input, filters, kernel_size, strides, padding, name, group)
    bn = tf.layers.batch_normalization(conv, training=is_train, name=name+'_bn')
    return tf.nn.relu(bn, name=name+'_relu')

def bn_relu_conv2d(is_train, input, filters, kernel_size, strides, padding, name):

    bn = tf.layers.batch_normalization(input, training=is_train, name=name+'_bn')
    relu = tf.nn.relu(bn, name=name+'_relu')
    return tf.layers.conv2d(relu, filters, kernel_size, strides, padding, name=name)

def get_key_feature(input, is_train, name):

    input_shape = input.get_shape().as_list()
    if len(input_shape) > 4:
        input = tf.reshape(input, [-1] + input_shape[2:])

    if config.use_fc_key:
        contrloller_input = bn_relu_conv2d(is_train, input, config.key_dim, config.slot_size[0:2], [1, 1], 'valid', name=name)
    else:
        contrloller_input = tf.layers.average_pooling2d(input, config.slot_size[0:2], [1, 1], 'valid', name=name)

    if len(input_shape) > 4:
        c_shape = contrloller_input.get_shape().as_list()
        contrloller_input = tf.reshape(contrloller_input, input_shape[0:2]+c_shape[1:])

    return contrloller_input
