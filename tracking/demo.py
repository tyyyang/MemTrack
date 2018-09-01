import tensorflow as tf
import time
import sys
sys.path.append('../')
import os
import json
import config
from tracking.tracker import Tracker, Model
from collections import OrderedDict
import cv2

class Sequence:

    def __init__(self, name, path, startFrame, endFrame, attributes,
        nz, ext, imgFormat, gtRect, init_rect):
        self.name = name
        self.path = path
        self.startFrame = startFrame
        self.endFrame = endFrame
        self.attributes = attributes
        self.nz = nz
        self.ext = ext
        self.imgFormat = imgFormat
        self.gtRect = gtRect
        self.init_rect = init_rect
        self.__dict__ = OrderedDict([
            ('name', self.name),
            ('path', self.path),
            ('startFrame', self.startFrame),
            ('endFrame', self.endFrame),
            ('attributes', self.attributes),
            ('nz', self.nz),
            ('ext', self.ext),
            ('imgFormat', self.imgFormat),
            ('init_rect', self.init_rect),
            ('gtRect', self.gtRect)])

def load_seq_config(name):
    if name == 'Jogging-1':
        seq_name = 'Jogging'
        config_name = 'cfg1.json'
    elif name == 'Jogging-2':
        seq_name = 'Jogging'
        config_name = 'cfg2.json'
    elif name == 'Skating2-1':
        seq_name = 'Skating2'
        config_name = 'cfg1.json'
    elif name == 'Skating2-2':
        seq_name = 'Skating2'
        config_name = 'cfg2.json'
    else:
        seq_name = name
        config_name = 'cfg.json'
    src = os.path.join(config.otb_data_dir,seq_name,config_name)
    configFile = open(src)
    string = configFile.read()
    j = json.loads(string)
    seq = Sequence(**j)
    seq.path = os.path.join(os.path.abspath(seq.path), '')

    seq.len = seq.endFrame - seq.startFrame + 1
    seq.s_frames = [None] * seq.len

    for i in range(seq.len):
        image_no = seq.startFrame + i
        _id = seq.imgFormat.format(image_no)
        seq.s_frames[i] = seq.path + _id

    seq.init_rect = seq.gtRect[0]
    return seq

def display_result(image, pred_boxes, frame_idx, seq_name=None):
    if len(image.shape) == 3:
        r, g, b = cv2.split(image)
        image = cv2.merge([b, g, r])
    pred_boxes = pred_boxes.astype(int)
    cv2.rectangle(image, tuple(pred_boxes[0:2]), tuple(pred_boxes[0:2] + pred_boxes[2:4]), (0, 0, 255), 2)

    cv2.putText(image, 'Frame: %d' % frame_idx, (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255))
    cv2.imshow('tracker', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return True
    if config.is_save:
        cv2.imwrite(os.path.join(config.save_path, seq_name, '%04d.jpg' % frame_idx), image)

def run_tracker():
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.Session(config=config_proto) as sess:
        os.chdir('../')
        model = Model(sess)
        tracker = Tracker(model)
        seq = load_seq_config('Basketball')
        bbox = seq.init_rect
        res = []
        res.append(bbox)
        start_time = time.time()
        tracker.initialize(seq.s_frames[0], bbox)

        for idx in range(1, len(seq.s_frames)):
            tracker.idx = idx
            bbox, cur_frame = tracker.track(seq.s_frames[idx])
            display_result(cur_frame, bbox, idx)
            res.append(bbox.tolist())
        end_time = time.time()
        type = 'rect'
        fps = idx/(end_time-start_time)

    return res, type, fps

if __name__ == '__main__':
    run_tracker()