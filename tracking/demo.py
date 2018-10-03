import tensorflow as tf
import time
import sys
sys.path.append('../')
import os
import config
from tracking.tracker import Tracker, Model
import cv2

def load_seq_config(seq_name):

    src = os.path.join(config.otb_data_dir,seq_name,'groundtruth_rect.txt')
    gt_file = open(src)
    lines = gt_file.readlines()
    gt_rects = []
    for gt_rect in lines:
        rect = [int(v) for v in gt_rect[:-1].split(',')]
        gt_rects.append(rect)

    init_rect= gt_rects[0]
    img_path = os.path.join(config.otb_data_dir,seq_name,'img')
    img_names = sorted(os.listdir(img_path))
    s_frames = [os.path.join(img_path, img_name) for img_name in img_names]

    return init_rect, s_frames

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
        init_rect, s_frames = load_seq_config('Basketball')
        bbox = init_rect
        res = []
        res.append(bbox)
        start_time = time.time()
        tracker.initialize(s_frames[0], bbox)

        for idx in range(1, len(s_frames)):
            tracker.idx = idx
            bbox, cur_frame = tracker.track(s_frames[idx])
            display_result(cur_frame, bbox, idx)
            res.append(bbox.tolist())
        end_time = time.time()
        type = 'rect'
        fps = idx/(end_time-start_time)

    return res, type, fps

if __name__ == '__main__':
    run_tracker()