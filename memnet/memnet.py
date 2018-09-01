# ------------------------------------------------------------------
# Tensorflow implementation of
#  "Learning Dynamic Memory Networks for Object Tracking", ECCV,2018
# Licensed under The MIT License [see LICENSE for details]
# Written by Tianyu Yang (tianyu-yang.com)
# ------------------------------------------------------------------

import collections
import tensorflow as tf
from memnet.access import MemoryAccess, AccessState
from feature import get_key_feature
import config

MemNetState = collections.namedtuple('MemNetState', ('controller_state', 'access_state'))

def attention(input, query, scope=None):

    input_shape = input.get_shape().as_list()
    input_transform = tf.layers.conv2d(input, input_shape[-1], [1, 1], [1, 1], use_bias=False, name='input_layer')
    query_transform = tf.layers.dense(query, input_shape[-1], name='query_layer')
    query_transform = tf.expand_dims(tf.expand_dims(query_transform, 1), 1)
    addition = tf.nn.tanh(input_transform + query_transform, name='addition')
    addition_transform = tf.layers.conv2d(addition, 1, [1, 1], [1, 1], use_bias=False, name='score')
    addition_shape = addition_transform.get_shape().as_list()
    score = tf.nn.softmax(tf.reshape(addition_transform, [addition_shape[0], -1]))

    if int(scope) < config.summary_display_step:
        max_idxes = tf.argmax(score, 1)
        tf.summary.histogram('max_idxes_{}'.format(scope),max_idxes)
        max_value = tf.reduce_max(score, 1)
        tf.summary.histogram('max_value_{}'.format(scope), max_value)

    score = tf.reshape(score, addition_shape)
    return tf.reduce_sum(input*score, [1,2]), score

class MemNet(tf.nn.rnn_cell.RNNCell):

    def __init__(self, hidden_size, memory_size, slot_size, is_train):
        super(MemNet, self).__init__()
        # self._controller = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        # if is_train and config.keep_prob < 1:
        #     self._controller = tf.nn.rnn_cell.DropoutWrapper(self._controller,
        #                                                      input_keep_prob=config.keep_prob,
        #                                                      output_keep_prob=config.keep_prob)
        keep_prob = config.keep_prob if is_train else 1.0
        self._controller = tf.contrib.rnn.LayerNormBasicLSTMCell(hidden_size, layer_norm=True, dropout_keep_prob=keep_prob)
        self._memory_access = MemoryAccess(memory_size, slot_size, is_train)
        self._hidden_size = hidden_size
        self._memory_size = memory_size
        self._slot_size = slot_size
        self._is_train = is_train

    def __call__(self, inputs, prev_state, scope=None):

        prev_controller_state = prev_state.controller_state
        prev_access_state = prev_state.access_state

        search_feature = inputs[0]
        memory_for_writing = inputs[1]

        # get lstm controller input
        controller_input = get_key_feature(search_feature, self._is_train, 'search_key')

        attention_input, self.att_score = attention(controller_input, prev_controller_state[1], scope)

        controller_output, controller_state = self._controller(attention_input, prev_controller_state, scope)

        access_inputs = (memory_for_writing, controller_output)
        access_output, access_state = self._memory_access(access_inputs, prev_access_state, scope)

        return access_output, MemNetState(access_state=access_state, controller_state=controller_state)

    def initial_state(self, init_feature):

        init_key = tf.squeeze(get_key_feature(init_feature, self._is_train, 'init_memory_key'), [1, 2])
        c_state = tf.layers.dense(init_key, self._hidden_size, activation=tf.nn.tanh, name='c_state')
        h_state = tf.layers.dense(init_key, self._hidden_size, activation=tf.nn.tanh, name='h_state')
        batch_size = init_key.get_shape().as_list()[0]
        controller_state = tf.nn.rnn_cell.LSTMStateTuple(c_state, h_state)
        write_weights = tf.one_hot([0]*batch_size, self._memory_size, axis=-1, dtype=tf.float32)
        read_weight = tf.zeros([batch_size, self._memory_size], tf.float32)
        control_factors = tf.one_hot([2]*batch_size, 3, axis=-1, dtype=tf.float32)
        write_decay = tf.zeros([batch_size, 1], tf.float32)
        usage = tf.one_hot([0]*batch_size, self._memory_size, axis=-1, dtype=tf.float32)
        memory = tf.zeros([batch_size, self._memory_size]+self._slot_size, tf.float32)
        access_state = AccessState(init_memory=init_feature,
                                   memory=memory,
                                   read_weight=read_weight,
                                   write_weight=write_weights,
                                   control_factors=control_factors,
                                   write_decay = write_decay,
                                   usage=usage)
        return MemNetState(controller_state=controller_state, access_state=access_state)

    @property
    def state_size(self):
        return MemNetState(controller_state=self._controller.state_size, access_state=self._memory_access.state_size)

    @property
    def output_size(self):
        return tf.TensorShape(self._slot_size)