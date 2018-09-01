# ------------------------------------------------------------------
# Tensorflow implementation of
#  "Learning Dynamic Memory Networks for Object Tracking", ECCV,2018
# Licensed under The MIT License [see LICENSE for details]
# Written by Tianyu Yang (tianyu-yang.com)
# ------------------------------------------------------------------

import tensorflow as tf
from model import ModeKeys
import config
import time
import os

class Estimator():

    def __init__(self, train_input_fn, eval_input_fn, model_fn):

        self._train_input_fn = train_input_fn
        self._eval_input_fn = eval_input_fn
        self._model_fn = model_fn
        tf.set_random_seed(1234)
        self._max_patience = 10 * config.validate_step
        self._best_value = None
        self._best_step = None
        self.build_eval()

    def train(self):
        config_proto = tf.ConfigProto()
        config_proto.gpu_options.allow_growth = True
        with tf.Graph().as_default(), tf.Session(config=config_proto) as sess:
            features, labels = self._train_input_fn()
            train_spec = self._model_fn(features, labels, ModeKeys.TRAIN)
            summary_writer = tf.summary.FileWriter(config.summaries_dir + 'train', sess.graph)

            global_step = tf.train.get_or_create_global_step()
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            enqueue_threads = tf.train.start_queue_runners(sess, coord=coord)

            idx = sess.run(global_step) + 1
            while not coord.should_stop() and idx <= config.max_iterations:
                start_time = time.time()

                if idx % config.summary_save_step == 0:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary, dist_error, loss, _ = sess.run(
                        [train_spec.summary, train_spec.dist_error, train_spec.loss, train_spec.train],
                        options=run_options,
                        run_metadata=run_metadata)

                    summary_writer.add_run_metadata(run_metadata, 'step%03d' % idx)
                    summary_writer.add_summary(summary, idx)
                    print('Adding run metadata for', idx)
                else:
                    dist_error, loss, _ = sess.run([train_spec.dist_error, train_spec.loss, train_spec.train])

                print("Step: %d Loss: %f, Dist error: %f  Speed: %.0f examples per second" %
                      (idx, loss, dist_error, config.batch_size * config.time_step / (time.time() - start_time)))

                if idx % config.model_save_step == 0 or idx == config.max_iterations or idx % config.validate_step == 0:
                    checkpoint_path = os.path.join(config.checkpoint_dir, 'model.ckpt')
                    train_spec.saver.save(sess, checkpoint_path, global_step=idx, write_meta_graph=False)
                    print('Save to checkpoint at step %d' % (idx))

                if idx % config.validate_step == 0:
                    if self.evaluate(idx, 'loss'):
                        coord.request_stop()

                idx = sess.run(tf.train.get_or_create_global_step()) + 1

            summary_writer.close()
            coord.join(enqueue_threads)

    def build_eval(self):

        with tf.Graph().as_default() as graph:
            features, labels = self._eval_input_fn()
            self._eval_spec = self._model_fn(features, labels, ModeKeys.EVAL)
            self._eval_summary_writer = tf.summary.FileWriter(config.summaries_dir + 'eval', graph)
            self._eval_graph = graph

    def evaluate(self, global_step, stop_metric='loss'):

        config_proto = tf.ConfigProto()
        config_proto.gpu_options.allow_growth = True
        with self._eval_graph.as_default(), tf.Session(config=config_proto) as sess:
            ckpt = tf.train.get_checkpoint_state(config.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                self._eval_spec.saver.restore(sess, ckpt.model_checkpoint_path)
                print('Checkpoint restored from %s' % (config.checkpoint_dir))

            coord = tf.train.Coordinator()
            enqueue_threads = tf.train.start_queue_runners(sess, coord=coord)

            totoal_dist_error = 0
            totoal_loss = 0
            i = 0
            print('Starting validate current network......')
            while i < config.max_iterations_eval:
                dist_error, loss = sess.run([self._eval_spec.dist_error, self._eval_spec.loss])
                totoal_dist_error += dist_error
                totoal_loss += loss
                i += 1
                print('Examples %d dist error: %f loss: %f' % (i, dist_error, loss))

            coord.request_stop()
            coord.join(enqueue_threads)
            avg_dist_error = totoal_dist_error / config.max_iterations_eval
            avg_loss = totoal_loss / config.max_iterations_eval
            print('val_dist_error: %f' % (avg_dist_error))
            print('val_loss: %f' % (avg_loss))

            summary = tf.Summary()
            # summary.ParseFromString(sess.run(self._eval_spec.summary))
            summary.value.add(tag='dist_error', simple_value=avg_dist_error)
            summary.value.add(tag='loss', simple_value=avg_loss)
            self._eval_summary_writer.add_summary(summary, global_step)

            coord.request_stop()
            coord.join(enqueue_threads)

        if stop_metric == 'loss':
            value = avg_loss
        elif stop_metric == 'dist_error':
            value = avg_dist_error
        else:
            value = avg_dist_error

        if (self._best_value is None) or \
            (value < self._best_value):
            self._best_value = value
            self._best_step = global_step

        should_stop = (global_step - self._best_step >= self._max_patience)
        if should_stop:
            print('Stopping... Best step: {} with {} = {}.' \
                .format(self._best_step, stop_metric, self._best_value))
        return should_stop

