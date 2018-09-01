# ------------------------------------------------------------------
# Tensorflow implementation of
#  "Learning Dynamic Memory Networks for Object Tracking", ECCV,2018
# Licensed under The MIT License [see LICENSE for details]
# Written by Tianyu Yang (tianyu-yang.com)
# ------------------------------------------------------------------

import collections
import tensorflow as tf
from memnet.addressing import cosine_similarity, attention_read, update_usage, calc_allocation_weight
from feature import get_key_feature
import config

AccessState = collections.namedtuple('AccessState', (
        'init_memory', 'memory', 'read_weight', 'write_weight', 'control_factors', 'write_decay', 'usage'))


def _reset_and_write(memory, write_weight, write_decay, control_factors, values):

    weight_shape = write_weight.get_shape().as_list()
    write_weight = tf.reshape(write_weight, weight_shape+[1,1,1])
    decay = write_decay*tf.expand_dims(control_factors[:, 1], 1) + tf.expand_dims(control_factors[:, 2], 1)
    decay_expand = tf.expand_dims(tf.expand_dims(tf.expand_dims(decay, 1), 2), 3)
    decay_weight = write_weight*decay_expand

    memory *= 1 - decay_weight
    values = tf.expand_dims(values, 1)
    memory += decay_weight * values

    return memory


class MemoryAccess(tf.nn.rnn_cell.RNNCell):

    def __init__(self, memory_size, slot_size, is_train):
        super(MemoryAccess, self).__init__()
        self._memory_size = memory_size
        self._slot_size = slot_size
        self._is_train = is_train


    def __call__(self, inputs, prev_state, scope=None):

        memory_for_writing = inputs[0]
        controller_output = inputs[1]
        read_key, read_strength, control_factors, write_decay, residual_vector = self._transform_input(controller_output)

        # Write previous template to memory.
        memory = _reset_and_write(prev_state.memory, prev_state.write_weight,
                                  prev_state.write_decay, prev_state.control_factors, memory_for_writing)

        # Read from memory.
        read_weight = self._read_weights(read_key, read_strength, memory)
        read_weight_expand = tf.reshape(read_weight, [-1, self._memory_size, 1, 1, 1])
        residual_vector = tf.reshape(residual_vector, [-1, 1, 1, 1, self._slot_size[2]])
        read_memory = tf.reduce_sum(residual_vector*read_weight_expand*memory, [1])

        # calculate the allocation weight
        allocation_weight = calc_allocation_weight(prev_state.usage, self._memory_size)

        # calculate the write weight for next frame writing
        write_weight = self._write_weights(control_factors, read_weight, allocation_weight)

        # update usage using read & write weights and previous usage
        usage = update_usage(write_weight, read_weight, prev_state.usage)

        # summary
        if int(scope) < config.summary_display_step:
            tf.summary.histogram('write_factor/{}'.format(scope), control_factors[:, 0])
            tf.summary.histogram('read_factor/{}'.format(scope), control_factors[:, 1])
            tf.summary.histogram('allocation_factor/{}'.format(scope), control_factors[:, 2])
            tf.summary.histogram('residual_vector/{}'.format(scope), residual_vector)
            tf.summary.histogram('write_decay/{}'.format(scope), write_decay)
            tf.summary.histogram('read_key/{}'.format(scope), read_key)
            if not config.use_attention_read:
                tf.summary.histogram('read_strength/{}'.format(scope), read_strength)

        return read_memory+prev_state.init_memory, AccessState(
            init_memory=prev_state.init_memory,
            memory=memory,
            write_weight=write_weight,
            read_weight=read_weight,
            control_factors=control_factors,
            write_decay=write_decay,
            usage=usage)

    def _transform_input(self, input):

        control_factors = tf.nn.softmax(tf.layers.dense(input, 3, name='control_factors'))
        write_decay = tf.sigmoid(tf.layers.dense(input, 1, name='write_decay'))
        residual_vector = tf.sigmoid(tf.layers.dense(input, self._slot_size[2], name='add_vector'))

        read_key = tf.layers.dense(input, config.key_dim, name='read_key')
        if config.use_attention_read:
            read_strength = None
        else:
            read_strength = tf.layers.dense(input, 1, bias_initializer=tf.ones_initializer(), name='write_strengths')

        return read_key, read_strength, control_factors, write_decay, residual_vector

    def _write_weights(self, control_factors, read_weight, allocation_weight):

        return tf.expand_dims(control_factors[:, 1], 1) * read_weight + tf.expand_dims(control_factors[:, 2], 1) * allocation_weight

    def _read_weights(self, read_key, read_strength, memory):

        memory_key = tf.squeeze(get_key_feature(memory, self._is_train, 'memory_key'),[2,3])
        if config.use_attention_read:
            return attention_read(read_key, memory_key)
        else:
            return cosine_similarity(memory_key, read_key, read_strength)


    @property
    def state_size(self):

        return AccessState(init_memory=tf.TensorShape([self._memory_size]+self._slot_size),
                memory=tf.TensorShape([self._memory_size]+self._slot_size),
                read_weight=tf.TensorShape([self._memory_size]),
                write_weight=tf.TensorShape([self._memory_size]),
                write_decay=tf.TensorShape([1]),
                control_factors=tf.TensorShape([3]),
                usage=tf.TensorShape([self._memory_size]))

    @property
    def output_size(self):

        return tf.TensorShape(self._slot_size)
