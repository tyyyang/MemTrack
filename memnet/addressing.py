# ------------------------------------------------------------------
# Tensorflow implementation of
#  "Learning Dynamic Memory Networks for Object Tracking", ECCV,2018
# Licensed under The MIT License [see LICENSE for details]
# Written by Tianyu Yang (tianyu-yang.com)
# ------------------------------------------------------------------

import config
import tensorflow as tf

# Ensure values are greater than epsilon to avoid numerical instability.
_EPSILON = 1e-6


def _vector_norms(m):

    squared_norms = tf.reduce_sum(m * m, axis=2, keep_dims=True)
    return tf.sqrt(squared_norms + _EPSILON)

def _weighted_softmax(activations, strengths, strengths_op):

    sharp_activations = activations * strengths_op(strengths)
    softmax_weights = tf.nn.softmax(sharp_activations)
    # softmax_weights = tf.nn.l2_normalize(sharp_activations, 1)
    return softmax_weights

def cosine_similarity(memory, keys, strengths, strength_op=tf.nn.softplus):

    # Calculates the inner product between the query vector and words in memory.
    keys = tf.expand_dims(keys, 1)
    dot = tf.matmul(keys, memory, adjoint_b=True)

    # Outer product to compute denominator (euclidean norm of query and memory).
    memory_norms = _vector_norms(memory)
    key_norms = _vector_norms(keys)
    norm = tf.matmul(key_norms, memory_norms, adjoint_b=True)

    # Calculates cosine similarity between the query vector and words in memory.
    similarity = dot / (norm + _EPSILON)

    return _weighted_softmax(tf.squeeze(similarity, [1]), strengths, strength_op)

def attention_read(read_key, memory_key):

    memory_key = tf.expand_dims(memory_key, 1)
    input_transform = tf.layers.conv2d(memory_key, 256, [1, 1], [1, 1], use_bias=False, name='memory_key_layer')
    query_transform = tf.layers.dense(read_key, 256, name='read_key_layer')
    query_transform = tf.expand_dims(tf.expand_dims(query_transform, 1), 1)
    addition = tf.nn.tanh(input_transform + query_transform, name='addition_layer')
    addition_transform = tf.layers.conv2d(addition, 1, [1, 1], [1, 1], use_bias=False, name='score_layer')
    addition_shape = addition_transform.get_shape().as_list()
    return tf.nn.softmax(tf.reshape(addition_transform, [addition_shape[0], -1]))

def update_usage(write_weights, read_weights, prev_usage):

    # write_weights = tf.stop_gradient(write_weights)
    # read_weights = tf.stop_gradient(read_weights)
    usage = config.usage_decay*prev_usage + write_weights + read_weights
    return usage

def calc_allocation_weight(usage, memory_size):

    usage = tf.stop_gradient(usage)
    nonusage = 1 - usage
    sorted_nonusage, indices = tf.nn.top_k(nonusage, k=1, name='sort')
    allocation_weights = tf.one_hot(tf.squeeze(indices, [1]), memory_size)

    return allocation_weights
