# ------------------------------------------------------------------
# Tensorflow implementation of
#  "Learning Dynamic Memory Networks for Object Tracking", ECCV,2018
# Licensed under The MIT License [see LICENSE for details]
# Written by Tianyu Yang (tianyu-yang.com)
# ------------------------------------------------------------------

import tensorflow as tf
from tensorflow.python.util import nest
from memnet.utils import weights_summay
import config

def rnn(cell, inputs, initial_state, scope=None):
    """Creates a recurrent neural network specified by RNNCell `cell`."""

    if not isinstance(cell, tf.contrib.rnn.RNNCell):
        raise TypeError("cell must be an instance of RNNCell")
    if not nest.is_sequence(inputs):
        raise TypeError("inputs must be a sequence")
    if not inputs:
        raise ValueError("inputs must not be empty")

    input_shape = inputs[0].get_shape().as_list()
    input_list = []
    for input in inputs:
        input_list.append([tf.squeeze(input_, [1])
                           for input_ in tf.split(axis=1, num_or_size_splits=input_shape[1], value=input)])

    num_input = len(inputs)
    inputs = []
    for i in range(input_shape[1]):
        inputs.append(tuple([input_list[j][i] for j in range(num_input)]))

    outputs = []
    states = []
    # Create a new scope in which the caching device is either
    # determined by the parent scope, or is set to place the cached
    # Variable using the same placement as for the rest of the RNN.
    with tf.variable_scope(scope or "RNN") as varscope:
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)

        state = initial_state

        for time, input_ in enumerate(inputs):
            if time > 0: varscope.reuse_variables()
            call_cell = lambda: cell(input_, state, str(time))
            output, state = call_cell()
            outputs.append(output)
            states.append(state)

    # summary for all these weights
    if len(inputs) >= config.summary_display_step:
        for i in range(config.summary_display_step):
            state = states[i]
            weights_summay(state.access_state.memory, 'memory_slot/{}'.format(i))
            weights_summay(state.access_state.read_weight, 'read_weight/{}'.format(i))
            weights_summay(state.access_state.write_weight, 'write_weight/{}'.format(i))
            weights_summay(state.access_state.usage, 'usage/{}'.format(i))
    output_shape = outputs[0].get_shape().as_list()
    outputs = tf.reshape(tf.concat(axis=1, values=outputs), [-1, input_shape[1]] + output_shape[1:])
    return (outputs, state)