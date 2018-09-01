# ------------------------------------------------------------------
# Tensorflow implementation of
#  "Learning Dynamic Memory Networks for Object Tracking", ECCV,2018
# Licensed under The MIT License [see LICENSE for details]
# Written by Tianyu Yang (tianyu-yang.com)
# ------------------------------------------------------------------

import tensorflow as tf

def weights_summay(weight, name):
    weight_shape = weight.get_shape().as_list()

    for i in range(weight_shape[1]):
        tf.summary.histogram('Memory_{}/{}'.format(i, name), weight[:,i])