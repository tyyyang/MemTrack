# ------------------------------------------------------------------
# Tensorflow implementation of
#  "Learning Dynamic Memory Networks for Object Tracking", ECCV,2018
# Licensed under The MIT License [see LICENSE for details]
# Written by Tianyu Yang (tianyu-yang.com)
# ------------------------------------------------------------------

import config
import tensorflow as tf
import numpy as np
from feature import extract_feature
from input import generate_labels_overlap, generate_labels_dist
from memnet.memnet import MemNet
from memnet.rnn import rnn
import collections

class ModeKeys():
  TRAIN = 'train'
  EVAL = 'eval'
  PREDICT = 'predict'

EstimatorSpec = collections.namedtuple('EstimatorSpec', ['predictions', 'loss', 'dist_error', 'train', 'summary', 'saver'])

def get_cnn_feature(input, reuse, mode):

    input_shape = input.get_shape().as_list()
    if len(input_shape) > 4:
        input = tf.reshape(input, [-1] + input_shape[2:])

    is_train = True if mode == ModeKeys.TRAIN else False
    with tf.variable_scope('feature_extraction', reuse=reuse):
        cnn_feature = extract_feature(is_train, input)

    if len(input_shape) > 4:
        cnn_feature_shape = cnn_feature.get_shape().as_list()
        cnn_feature = tf.reshape(cnn_feature, input_shape[0:2]+cnn_feature_shape[1:])

    return cnn_feature

def batch_conv(A, B, mode):

    a_shape = A.get_shape().as_list()
    if len(a_shape) > 4:
        A = tf.reshape(A, [-1] + a_shape[2:])
    b_shape = B.get_shape().as_list()
    if len(b_shape) > 4:
        B = tf.reshape(B, [-1] + b_shape[2:])
    batch_size = A.get_shape().as_list()[0]

    output = tf.map_fn(lambda inputs: tf.nn.conv2d(tf.expand_dims(inputs[0], 0), tf.expand_dims(inputs[1], 3), [1,1,1,1], 'VALID'),
                       elems=[A, B],
                       dtype=tf.float32,
                       parallel_iterations=batch_size)
    is_train = True if mode == ModeKeys.TRAIN else False
    output = tf.layers.batch_normalization(tf.squeeze(output, [1]), training=is_train, name='bn_response')
    return tf.squeeze(output, [3])

def get_predictions(query_feature, search_feature, mode):

    with tf.variable_scope('mann'):
       mann_cell = MemNet(config.hidden_size, config.memory_size, config.slot_size, True)

    initial_state = mann_cell.initial_state(query_feature[:, 0])

    inputs = (search_feature, query_feature)
    outputs, final_state = rnn(cell=mann_cell, inputs=inputs, initial_state=initial_state)

    response = batch_conv(search_feature, outputs, mode)

    return response


def focal_loss(labels, predictions, gamma=2, epsilon=1e-7, scope=None):

    with tf.name_scope(scope, "focal_loss", (predictions, labels)) as scope:
        predictions = tf.to_float(predictions)
        labels = tf.to_float(labels)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        preds = tf.where(
            tf.equal(labels, 1), predictions, 1. - predictions)
        losses = -(1. - preds) ** gamma * tf.log(preds + epsilon)
        return losses

def get_loss(outputs, labels, mode):

    if mode == tf.estimator.ModeKeys.PREDICT:
        return None
    outputs_shape = outputs.get_shape().as_list()
    if config.label_type == 0:
        labels_response, weights = generate_labels_overlap(np.array(outputs_shape[1:3]), labels)
    else:
        labels_response, weights = generate_labels_dist(outputs_shape[0], np.array(outputs_shape[1:3]))
    if config.use_focal_loss:
        loss = tf.reduce_sum(weights * focal_loss(labels=labels_response, predictions=tf.nn.sigmoid(outputs))) / outputs_shape[0]
    else:
        loss = tf.reduce_sum(weights*tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_response, logits=outputs))/outputs_shape[0]
    tf.summary.scalar('loss', loss)
    return loss

def get_dist_error(outputs, mode):

    if mode == tf.estimator.ModeKeys.PREDICT:
        return None
    outputs_shape = outputs.get_shape().as_list()
    outputs = tf.reshape(outputs, [outputs_shape[0], -1])
    pred_loc_idx = tf.argmax(outputs, 1)
    loc_x = pred_loc_idx%outputs_shape[1]
    loc_y = pred_loc_idx//outputs_shape[1]
    pred_loc = tf.stack([loc_x, loc_y], 1)
    gt_loc = tf.tile(tf.expand_dims([outputs_shape[1]/2, outputs_shape[1]/2], 0), [outputs_shape[0], 1])
    dist_error = tf.losses.mean_squared_error(predictions=pred_loc, labels=gt_loc)
    tf.summary.scalar('dist_error', dist_error)
    return dist_error

def get_train_op(loss, mode):

    if mode != ModeKeys.TRAIN:
        return None

    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(config.learning_rate, global_step, config.decay_circles, config.lr_decay, staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)

    tvars = tf.trainable_variables()
    regularizer = tf.contrib.layers.l2_regularizer(config.weight_decay)
    regularizer_loss = tf.contrib.layers.apply_regularization(regularizer, tvars)
    loss += regularizer_loss
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), config.clip_gradients)
    # optimizer = tf.train.GradientDescentOptimizer(self.lr)
    optimizer = tf.train.AdamOptimizer(learning_rate)

    batchnorm_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(batchnorm_update_ops):
        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step)

    return train_op

def get_summary(mode):

    if mode == ModeKeys.PREDICT:
        return None
    return tf.summary.merge_all()

def get_saver():

    return tf.train.Saver(tf.global_variables(), max_to_keep=15)

def model_fn(features, labels, mode):
    # get cnn feature for query and search
    query_feature = get_cnn_feature(features['query'], None, mode)
    search_feature = get_cnn_feature(features['search'], True, mode)

    predictions = get_predictions(query_feature, search_feature, mode)
    loss = get_loss(predictions, labels, mode)
    dist_error = get_dist_error(predictions, mode)
    train_op = get_train_op(loss, mode)
    summary = get_summary(mode)
    saver = get_saver()

    return EstimatorSpec(predictions, loss, dist_error, train_op, summary, saver)

def build_initial_state(init_query, mem_cell, mode):

    query_feature = get_cnn_feature(init_query, None, mode)
    return mem_cell.initial_state(query_feature[:,0])

def build_model(query, search, mem_cell, initial_state, mode):
    # get cnn feature for query and search
    query_feature = get_cnn_feature(query, True, mode)
    search_feature = get_cnn_feature(search, True, mode)

    inputs = (search_feature, query_feature)
    outputs, final_state = rnn(cell=mem_cell, inputs=inputs, initial_state=initial_state)

    response = batch_conv(search_feature, outputs, mode)
    saver = get_saver()

    return response, saver, final_state


if __name__=='__main__':
    query_patch = tf.placeholder(tf.float32, [10, 5, config.z_exemplar_size, config.z_exemplar_size, 3])
    search_patch = tf.placeholder(tf.float32, [10, 5, config.x_instance_size, config.x_instance_size, 3])
    features = {
        'query': query_patch,
        'search': search_patch
    }
    labels = tf.placeholder(tf.float32, [10, 5, 4])
    mode = ModeKeys.TRAIN

    esti_spec = model_fn(features, labels, mode)
    pass
