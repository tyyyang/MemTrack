# ------------------------------------------------------------------
# Tensorflow implementation of
#  "Learning Dynamic Memory Networks for Object Tracking", ECCV,2018
# Licensed under The MIT License [see LICENSE for details]
# Written by Tianyu Yang (tianyu-yang.com)
# ------------------------------------------------------------------

import glob
import os
import time
import numpy as np
import tensorflow as tf
import config

DEBUG = False

def generate_input_fn(is_train, tfrecords_path, batch_size, time_step):
    "Return _input_fn for use with Experiment."
    def _input_fn():

        with tf.device('/cpu:0'):
            query_patch, search_patch, bbox = _batch_input(is_train, tfrecords_path, batch_size, time_step)

            patches = {
                'query': query_patch,
                'search': search_patch,
            }
            return patches, bbox

    return _input_fn

def _batch_input(is_train, tfrecords_path, batch_size, time_step):

    if is_train:
        tf_files = glob.glob(os.path.join(tfrecords_path, 'train-*.tfrecords'))
        filename_queue = tf.train.string_input_producer(tf_files, shuffle=True, capacity=16)

        min_queue_examples = config.min_queue_examples
        examples_queue = tf.RandomShuffleQueue(
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples,
            dtypes=[tf.string])
        enqueue_ops = []
        for _ in range(config.num_readers):
            _, value = tf.TFRecordReader().read(filename_queue)
            enqueue_ops.append(examples_queue.enqueue([value]))

        tf.train.add_queue_runner(
            tf.train.QueueRunner(examples_queue, enqueue_ops))
        example_serialized = examples_queue.dequeue()
    else:
        tf_files = sorted(glob.glob(os.path.join(tfrecords_path, 'val-*.tfrecords')))
        filename_queue = tf.train.string_input_producer(tf_files, shuffle=False, capacity=8)
        _, example_serialized = tf.TFRecordReader().read(filename_queue)
        # example_serialized = next(tf.python_io.tf_record_iterator(self._tf_files[0]))
    images_and_labels = []
    for thread_id in range(config.num_preprocess_threads):
        sequence, context = _parse_example_proto(example_serialized)
        image_buffers = sequence['images']
        bboxes = sequence['bboxes']
        seq_len = tf.cast(context['seq_len'][0], tf.int32)
        z_exemplars, x_crops, y_crops = _process_images(image_buffers, bboxes, seq_len, thread_id, time_step, is_train)
        images_and_labels.append([z_exemplars, x_crops, y_crops])

    batch_z, batch_x, batch_y = tf.train.batch_join(images_and_labels,
                                                    batch_size=batch_size,
                                                    capacity=2 * config.num_preprocess_threads * batch_size)
    if is_train:
        tf.summary.image('exemplars', batch_z[0], 5)
        tf.summary.image('crops', batch_x[0], 5)

    return batch_z, batch_x, batch_y

def _process_images(image_buffers, bboxes, seq_len, thread_id, time_step, is_train):
    if config.is_limit_search:
        search_range = tf.minimum(config.max_search_range, seq_len - 1)
    else:
        search_range = seq_len-1
    rand_start_idx = tf.random_uniform([], 0, seq_len-search_range, dtype=tf.int32)
    selected_len = time_step + 1
    if is_train:
        frame_idxes = tf.range(rand_start_idx, rand_start_idx+search_range)
        shuffle_idxes = tf.random_shuffle(frame_idxes)
        selected_idxes = shuffle_idxes[0:selected_len]
        selected_idxes, _ = tf.nn.top_k(selected_idxes, selected_len)
        selected_idxes = selected_idxes[::-1]
    else:
        selected_idxes = tf.to_int32(tf.linspace(0.0, tf.to_float(seq_len - 1), selected_len))
    # self.seq_len = seq_len
    # self.search_range = search_range
    # self.selected_idxes = selected_idxes
    z_exemplars, y_exemplars, x_crops, y_crops = [], [], [], []
    shift = int((config.patch_size - config.z_exemplar_size) / 2)
    for i in range(selected_len):
        idx = selected_idxes[i]
        image_buffer = tf.gather(image_buffers, idx)
        image = tf.image.decode_jpeg(image_buffer, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image.set_shape([config.patch_size, config.patch_size, 3])

        # # Randomly distort the colors.
        # if is_train:
        #     image = _distort_color(image, thread_id)

        if i < time_step:
            # if self._is_train:
            exemplar = tf.image.crop_to_bounding_box(image, shift, shift, config.z_exemplar_size,
                                                     config.z_exemplar_size)
            if config.is_augment and i > 0:
                exemplar = _translate_and_strech(image,
                                                     [config.z_exemplar_size, config.z_exemplar_size],
                                                     config.max_strech_z, config.max_translate_z)
            z_exemplars.append(exemplar)
        if i > 0:
            bbox = tf.gather(bboxes, idx)
            if config.is_augment:
                image, bbox = _translate_and_strech(image, [config.x_instance_size, config.x_instance_size],
                                                        config.max_strech_x, config.max_translate_x, bbox)
            x_crops.append(image)
            y_crops.append(bbox)
    x_crops = tf.stack(x_crops, 0)
    y_crops = tf.stack(y_crops, 0)
    z_exemplars = tf.stack(z_exemplars, 0)
    return z_exemplars, x_crops, y_crops

def _translate_and_strech(image, m_sz, max_strech, max_translate=None, bbox=None, rgb_variance=None):

    m_sz_f = tf.convert_to_tensor(m_sz, dtype=tf.float32)
    img_sz = tf.convert_to_tensor(image.get_shape().as_list()[0:2],dtype=tf.float32)
    scale = 1+max_strech*tf.random_uniform([2], -1, 1, dtype=tf.float32)
    scale_sz = tf.round(tf.minimum(scale*m_sz_f, img_sz))

    if max_translate is None:
        shift_range = (img_sz - scale_sz) / 2
    else:
        shift_range = tf.minimum(float(max_translate), (img_sz-scale_sz)/2)

    start = (img_sz - scale_sz)/2
    shift_row = start[0] + tf.random_uniform([1], -shift_range[0], shift_range[0], dtype=tf.float32)
    shift_col = start[1] + tf.random_uniform([1], -shift_range[1], shift_range[1], dtype=tf.float32)

    x1 = shift_col/(img_sz[1]-1)
    y1 = shift_row/(img_sz[0]-1)
    x2 = (shift_col + scale_sz[1]-1)/(img_sz[1]-1)
    y2 = (shift_row + scale_sz[0]-1)/(img_sz[0]-1)
    crop_img = tf.image.crop_and_resize(tf.expand_dims(image,0),
                                        tf.expand_dims(tf.concat(axis=0, values=[y1, x1, y2, x2]), 0),
                                        [0], m_sz)
    crop_img = tf.squeeze(crop_img)
    if rgb_variance is not None:
        crop_img = crop_img + rgb_variance*tf.random_normal([1,1,3])

    if bbox is not None:
        new_bbox = bbox - tf.concat(axis=0, values=[shift_col, shift_row, shift_col, shift_row])
        scale_ratio = m_sz_f/tf.reverse(scale_sz, [0])
        new_bbox = new_bbox*tf.tile(scale_ratio,[2])
        return crop_img, new_bbox
    else:
        return crop_img

def _distort_color(image, thread_id=0):
    """Distort the color of the image.
    """
    color_ordering = thread_id % 2

    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)

    # The random_* ops do not necessarily clamp.
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image

def _parse_example_proto(example_serialized):

    context_features = {
        'seq_name': tf.FixedLenFeature([], dtype=tf.string),
        'seq_len': tf.FixedLenFeature(1, dtype=tf.int64),
        'trackid': tf.FixedLenFeature(1, dtype=tf.int64),
    }
    sequence_features = {
        'images': tf.FixedLenSequenceFeature([],dtype=tf.string),
        'bboxes': tf.FixedLenSequenceFeature([4],dtype=tf.float32)
    }
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(example_serialized, context_features, sequence_features)

    return sequence_parsed, context_parsed


def generate_labels_dist(batch_size, feat_size):
    dist = lambda i,j,orgin: np.linalg.norm(np.array([i,j])-orgin)
    labels = -np.ones(feat_size, dtype=np.int32)
    orgin = (np.array(feat_size) -1)/2
    for i in range(feat_size[0]):
        for j in range(feat_size[1]):
            distance = dist(i,j,orgin)
            if distance <= config.dist_thre:
                labels[i,j] = 1
            else:
                labels[i,j] = 0
    num_pos = np.count_nonzero(labels == 1)
    num_neg = np.count_nonzero(labels == 0)
    weights = np.zeros(feat_size, dtype=np.float32)
    weights[labels==1] = 0.5/num_pos
    weights[labels==0] = 0.5/num_neg
    batch_labels = np.tile(labels, [batch_size, 1, 1])
    batch_weights = np.tile(weights, [batch_size, 1, 1])
    return tf.convert_to_tensor(batch_labels, tf.float32), tf.convert_to_tensor(batch_weights)

def generate_labels_overlap(feat_size, bboxes, neg_flag=0):
    bboxes = tf.reshape(bboxes, [-1, 4])
    batch_labels, batch_weights = \
        tf.py_func(_generate_labels_overlap_py,
                   [feat_size, bboxes, (feat_size - 1)/2, neg_flag],
                   [tf.float32, tf.float32])
    bboxes_shape = bboxes.get_shape().as_list()
    batch_labels.set_shape([bboxes_shape[0]]+feat_size.tolist())
    batch_weights.set_shape([bboxes_shape[0]]+feat_size.tolist())
    return batch_labels, batch_weights

def _generate_labels_overlap_py(feat_size, y_crops, orgin, neg_flag=0):
    orig_size = feat_size*config.stride
    x = np.arange(0, orig_size[0], config.stride)+config.stride/2
    y = np.arange(0, orig_size[1], config.stride)+config.stride/2
    x, y = np.meshgrid(x, y)
    orgin = orgin*config.stride + config.stride/2
    batch_labels, batch_weights, batch_keep  = [], [], []
    for gt_bb_cur in y_crops:
        gt_size_cur = gt_bb_cur[2:4] - gt_bb_cur[0:2] + 1
        gt_bb_cur_new = np.hstack([orgin - (gt_size_cur - 1) / 2, orgin + (gt_size_cur - 1) / 2])
        sample_centers = np.vstack([x.ravel(), y.ravel(), x.ravel(), y.ravel()]).transpose()
        sample_bboxes = sample_centers + np.hstack([-(gt_size_cur-1)/2, (gt_size_cur-1)/2])

        overlaps = _bbox_overlaps(sample_bboxes, gt_bb_cur_new)

        pos_idxes = overlaps > config.overlap_thres
        neg_idxes = overlaps < config.overlap_thres
        labels = -np.ones(np.prod(feat_size), dtype=np.float32)
        labels[pos_idxes] = 1
        labels[neg_idxes] = neg_flag
        labels = np.reshape(labels, feat_size)

        num_pos = np.count_nonzero(labels == 1)
        num_neg = np.count_nonzero(labels == neg_flag)

        if DEBUG:
            print(gt_bb_cur)
            print((gt_bb_cur[0:2]+gt_bb_cur[2:4])/2)
            print('Positive samples:', num_pos, 'Negative samples:', num_neg)
            from matplotlib import pyplot as plt
            plt.imshow(labels)
            # # plt.imshow(np.reshape(overlaps, feat_size))
            plt.pause(1)

        weights = np.zeros(feat_size, dtype=np.float32)
        if num_pos != 0:
            weights[labels == 1] = 0.5 / num_pos
        if num_neg != 0:
            weights[labels == neg_flag] = 0.5 / num_neg
        batch_weights.append(np.expand_dims(weights, 0))
        batch_labels.append(np.expand_dims(labels, 0))

    batch_labels = np.concatenate(batch_labels, 0)
    batch_weights = np.concatenate(batch_weights, 0)
    return batch_labels, batch_weights

def _bbox_overlaps(sample_bboxes, gt_bbox):
    lt = np.maximum(sample_bboxes[:, 0:2], gt_bbox[0:2])
    rb = np.minimum(sample_bboxes[:, 2:4], gt_bbox[2:4])
    inter_area = np.maximum(rb - lt + 1, 0)
    inter_area = np.prod(inter_area, 1)
    union_area = np.prod(sample_bboxes[:, 2:4] - sample_bboxes[:, 0:2] + 1, 1) + np.prod(gt_bbox[2:4]-gt_bbox[0:2]+1, 0) - inter_area
    return inter_area / union_area


if __name__=='__main__':
    feat_size = np.array([15,15])
    label = np.array([[75,75,149,149]])
    label, weights = _generate_labels_overlap_py(feat_size, label, (feat_size - 1)/2)
    DEBUG = True
    if DEBUG:
        from utils.display_utils import display_train_input

    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.Session(config=config_proto) as sess:
        batch_size = 8
        time_steps = 4
        data_input_fn = generate_input_fn(True, config.tfrecords_path, batch_size, time_steps)
        batch_patch_op, batch_y_op = data_input_fn()

        coord = tf.train.Coordinator()
        enqueue_threads = tf.train.start_queue_runners(sess, coord=coord)

        while True:
            if coord.should_stop():
                break
            start_time = time.time()
            # seq_len, search_range, select_idxes =  sess.run([data_input.seq_len, data_input.search_range, data_input.selected_idxes])
            batch_patches, batch_y = sess.run([batch_patch_op, batch_y_op])
            end_time = time.time()
            print('cost time: %f' % (end_time - start_time))
            if DEBUG:
                batch_z = batch_patches['query']
                batch_x = batch_patches['search']
                if not display_train_input(batch_z, batch_x, batch_y):
                    break
                    # print(seq_len, search_range, select_idxes)
        coord.request_stop()
        coord.join(enqueue_threads)

