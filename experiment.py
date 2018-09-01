# ------------------------------------------------------------------
# Tensorflow implementation of
#  "Learning Dynamic Memory Networks for Object Tracking", ECCV,2018
# Licensed under The MIT License [see LICENSE for details]
# Written by Tianyu Yang (tianyu-yang.com)
# ------------------------------------------------------------------

import tensorflow as tf
from input import generate_input_fn
from model import model_fn
from estimator import Estimator
import config


def experiment():

    train_input_fn = generate_input_fn(
        is_train=True,
        tfrecords_path=config.tfrecords_path,
        batch_size=config.batch_size,
        time_step=config.time_step)

    eval_input_fn = generate_input_fn(
        is_train=False,
        tfrecords_path=config.tfrecords_path,
        batch_size=config.batch_size_eval,
        time_step=config.time_step_eval)

    estimator = Estimator(
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        model_fn=model_fn)

    estimator.train()


if __name__ == '__main__':
    if tf.gfile.Exists(config.summaries_dir):
        tf.gfile.DeleteRecursively(config.summaries_dir)
    tf.gfile.MakeDirs(config.summaries_dir)

    experiment()