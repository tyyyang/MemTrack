# ------------------------------------------------------------------
# Tensorflow implementation of
#  "Learning Dynamic Memory Networks for Object Tracking", ECCV,2018
# Licensed under The MIT License [see LICENSE for details]
# Written by Tianyu Yang (tianyu-yang.com)
# ------------------------------------------------------------------

import pickle
from PIL import Image, ImageDraw
import numpy as np
import os
import tensorflow as tf
import threading
from datetime import datetime
import sys
sys.path.append('../')
import config
import time

class Vid():
    pass

class EncodeJpeg():
    def __init__(self, sess):
        self.img = tf.placeholder(dtype=tf.uint8)
        self.img_buffer = tf.image.encode_jpeg(self.img)
        self.sess = sess
    def get_img_buffer(self, img):
        img_buffer = self.sess.run(self.img_buffer, {self.img: img})
        return img_buffer

def partition_vid(f_vid_list, num_threads):
    """To make it evenly distributed"""
    f_vid_list = sorted(f_vid_list, key=lambda x: x.n_frame)
    n_vid = len(f_vid_list)
    part_num = int(n_vid / num_threads)
    remain_num = n_vid - part_num * num_threads

    ranges_list = [[] for _ in range(num_threads)]
    for i in range(num_threads):
        for j in range(part_num):
            if j < int(part_num / 2):
                ranges_list[i].append(j * num_threads + i + remain_num)
            else:
                j_rev = part_num - (j - int(part_num / 2))
                ranges_list[i].append(j_rev * num_threads - i - 1 + remain_num)
    for i in range(remain_num):
        ranges_list[i].append(i)

    # Launch a thread for each spacing.
    print('Launching %d threads' % (num_threads))

    ranges_list_flat = sum(ranges_list, [])
    unique_value = np.unique(np.array(ranges_list_flat))
    assert len(unique_value) == n_vid
    frames = []
    for ranges in ranges_list:
        frame_sum = 0
        for x in ranges:
            frame_sum += f_vid_list[x].n_frame
        frames.append(frame_sum)

    print('Total frames for each threads\n', frames)
    return ranges_list

def build_tfrecords(vidb_f, data_path, num_threads, encode_jpeg):
    vidb = pickle.load(open(vidb_f, 'rb'))

    n_vid = len(vidb)
    part_idexes = partition_vid(vidb, num_threads)

    sys.stdout.flush()
    coord = tf.train.Coordinator()

    threads = []
    data_name = os.path.basename(data_path)
    if not os.path.exists(config.tfrecords_path):
        os.makedirs(config.tfrecords_path)

    for i, thread_idxes in enumerate(part_idexes):
        output_filename = '%s-%.3d-of-%.3d.tfrecords' % (data_name, i, num_threads)
        output_file = os.path.join(config.tfrecords_path, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)
        args = (thread_idxes, vidb, data_path, writer, encode_jpeg)
        t = threading.Thread(target=process_videos, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d videos in data set which contains %d valid objects in total.' %
          (datetime.now(), n_vid, n_valid_objs))
    sys.stdout.flush()


def process_videos_area(video_idxes, vidb, data_path, writer, encode_jpeg):
    for idx in video_idxes:
        one_video = vidb[idx]
        img_buffers_ids = [[] for _ in range(config.max_trackid)]
        bboxes_ids = [[] for _ in range(config.max_trackid)]
        occupy_area_ids = [[] for _ in range(config.max_trackid)]
        vid_name = one_video.dir
        for objs in one_video.objs:
            sum_area = 0
            len_area = 0
            for j, obj in enumerate(objs):
                if obj is not None:
                    occupy_area = (obj.xmax - obj.xmin) * (obj.ymax - obj.ymin) / (obj.width * obj.height)
                    sum_area += occupy_area
                    len_area += 1
                    roi_patch, bbox = process(obj, data_path)
                    img_buffer = encode_jpeg.get_img_buffer(roi_patch)
                    img_buffers_ids[j].append(img_buffer)
                    bboxes_ids[j].append(bbox.tolist())
                    occupy_area_ids[j].append(occupy_area)

        for id, area_id in enumerate(occupy_area_ids):
            sum_area = 0
            len_area = 0
            for area in area_id:
                sum_area += area
                len_area += 1
            if len_area>0:
                avg_area = sum_area / len_area
            else:
                avg_area = 0

            if avg_area > 0.25:
                img_buffers_ids[id] = []
                bboxes_ids[id] = []

        valid_objs = save_to_tfrecords(img_buffers_ids, bboxes_ids, vid_name, writer)
        # if is_train:
        #     valid_objs = save_to_tfrecords(img_buffers_ids, bboxes_ids, vid_name, writer, is_train)
        # else:
        #     valid_objs = save_to_tfrecords_eval(img_buffers_ids, bboxes_ids, vid_name, writer)
        lock.acquire()
        global n_valid_objs
        n_valid_objs += valid_objs
        lock.release()
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
              'Finish processing video: %s video id: %4d, record %2d valid objects' % (
              vid_name, one_video.id, valid_objs))

def process_videos(video_idxes, vidb, data_path, writer, encode_jpeg):
    for idx in video_idxes:
        one_video = vidb[idx]
        img_buffers_ids = [[] for _ in range(config.max_trackid)]
        bboxes_ids = [[] for _ in range(config.max_trackid)]
        vid_name = one_video.dir
        for objs in one_video.objs:
            for j, obj in enumerate(objs):
                if obj is not None:
                    roi_patch, bbox = process(obj, data_path)
                    img_buffer = encode_jpeg.get_img_buffer(roi_patch)
                    img_buffers_ids[j].append(img_buffer)
                    bboxes_ids[j].append(bbox.tolist())
        valid_objs = save_to_tfrecords(img_buffers_ids, bboxes_ids, vid_name, writer)
        # if is_train:
        #     valid_objs = save_to_tfrecords(img_buffers_ids, bboxes_ids, vid_name, writer, is_train)
        # else:
        #     valid_objs = save_to_tfrecords_eval(img_buffers_ids, bboxes_ids, vid_name, writer)
        lock.acquire()
        global n_valid_objs
        n_valid_objs += valid_objs
        lock.release()
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
              'Finish processing video: %s video id: %4d, record %2d valid objects' %(vid_name, one_video.id, valid_objs))

def process(obj, data_path):
    img = Image.open(os.path.join(data_path, obj.img_path))
    avg_color = tuple(np.mean(np.array(img), (0,1)).astype(int))

    # calculate roi region
    bb_c = np.array([(obj.xmin + obj.xmax)/2, (obj.ymin + obj.ymax)/2])
    bb_size = np.array([obj.xmax - obj.xmin + 1, obj.ymax - obj.ymin + 1])
    if config.fix_aspect:
        extend_size = bb_size + config.context_amount * (bb_size[0] + bb_size[1])
        z_size = np.sqrt(np.prod(extend_size))
    else:
        z_size = bb_size * config.z_scale
    z_scale = config.z_exemplar_size / z_size
    delta_size = config.patch_size - config.z_exemplar_size
    x_size = delta_size / z_scale + z_size
    x_sclae = config.patch_size / x_size
    roi = np.floor(np.concatenate([bb_c - (x_size - 1) / 2, bb_c + (x_size - 1) / 2], 0)).astype(int)

    bbox = np.array([obj.xmin, obj.ymin, obj.xmax, obj.ymax], dtype=np.float32)

    # calculate the image padding
    img_size = np.array([img.width, img.height])
    img_pad_xymin = np.maximum(0, -roi[0:2])
    img_pad_xymax = np.maximum(0, roi[2:4]-img_size+1)

    # if need padding
    if np.any(img_pad_xymin) or np.any(img_pad_xymax):
        pad_img_size = img_pad_xymax+img_pad_xymin+img_size
        img_pad = Image.new(img.mode, tuple(pad_img_size), avg_color)
        img_pad.paste(img, tuple(img_pad_xymin))

        # shift roi coordinate
        shift_xy = np.tile(img_pad_xymin, 2)
        roi += shift_xy
        bbox += shift_xy
    else:
        img_pad = img

    roi_patch = img_pad.crop(roi)
    roi_patch = roi_patch.resize([config.patch_size, config.patch_size])

    # shift bbox relative to roi
    roi_patch_size = np.array([config.patch_size, config.patch_size])
    bb_c_on_roi = (roi_patch_size - 1) / 2
    bb_size_on_roi = np.floor(bb_size * x_sclae)
    bbox = np.hstack([bb_c_on_roi - (bb_size_on_roi-1)/2, bb_c_on_roi + (bb_size_on_roi-1)/2])

    # img_draw = ImageDraw.Draw(roi_patch)
    # img_draw.rectangle(bbox.tolist(), outline=(255,0,0))
    # roi_patch.show()
    return np.array(roi_patch), bbox

def save_to_tfrecords(img_buffers_ids, bboxes_ids, seq_name, writer):

    valid_objs = 0
    for id, (imgs_per_id, bboxes_per_id) in enumerate(zip(img_buffers_ids, bboxes_ids)):
        seq_len = len(imgs_per_id)

        if seq_len < config.min_frames:
            continue
        valid_objs += 1
        example = convert_to_example(imgs_per_id, bboxes_per_id, seq_name, seq_len, id)
        writer.write(example.SerializeToString())
    return valid_objs

def convert_to_example(img_buffers, bboxes, seq_name, seq_len, trackid):

    context = tf.train.Features(feature={
        'seq_name': _bytes_feature(seq_name.encode('utf-8')),
        'seq_len': _int64_feature(seq_len),
        'trackid': _int64_feature(trackid)
    })
    feature_lists = tf.train.FeatureLists(feature_list={
        'images': _bytes_feature_list(img_buffers),
        'bboxes': _float_feature_list(bboxes)
    })
    example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)

    return example

def _int64_feature(value):
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature_list(values):
  return tf.train.FeatureList(feature=[_float_feature(v) for v in values])

def _bytes_feature_list(values):
  return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])


if __name__=='__main__':

    lock = threading.Lock()

    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.Session(config=config_proto) as sess:
        encode_jpeg = EncodeJpeg(sess)
        global n_valid_objs
        n_valid_objs = 0
        print('Start building training tfrecords.......')
        t_start = time.time()
        build_tfrecords(config.vidb_t, config.data_path_t, config.num_threads_t, encode_jpeg)
        t_end = time.time()
        print('The time for building training tfrecords is %f seconds:' % (t_end - t_start))

        n_valid_objs = 0
        print('Start building validation tfrecords.......')
        t_start = time.time()
        build_tfrecords(config.vidb_v, config.data_path_v, config.num_threads_v, encode_jpeg)
        t_end = time.time()
        print('The time for building validation tfrecords is %f seconds:' % (t_end - t_start))