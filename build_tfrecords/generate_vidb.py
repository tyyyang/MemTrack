# ------------------------------------------------------------------
# Tensorflow implementation of
#  "Learning Dynamic Memory Networks for Object Tracking", ECCV,2018
# Licensed under The MIT License [see LICENSE for details]
# Written by Tianyu Yang (tianyu-yang.com)
# ------------------------------------------------------------------

import sys
sys.path.append('../')
import os
from build_tfrecords.process_xml import process_xml
import pickle
import time
from datetime import datetime
import config

class Vid(object):
    pass

def generate_vidb(root_anno_path, vid_info_path, vidb_f):

    f_vid = open(vid_info_path)
    vid_info = f_vid.read().split('\n')
    if vid_info[-1] == '':
        vid_info.pop()

    vidb = []
    object_num = 0
    frame_num = 0
    for line in vid_info:
        frame_info = line.split(' ')
        vid_dir = frame_info[0]
        vid_id = int(frame_info[1])
        vid_n_frame = int(frame_info[2])
        vid = Vid()
        anno_path = os.path.join(root_anno_path, vid_dir)
        xml_files = sorted(os.listdir(anno_path))
        objs_one_video = []
        for xml_file in xml_files:
            frame_num += 1
            bboxes = process_xml(os.path.join(anno_path, xml_file))
            objs_one_frame = config.max_trackid * [None]
            for obj in bboxes:
                object_num += 1
                id = obj.trackid
                if id >= config.max_trackid:
                    print(obj.img_path)
                objs_one_frame[id] = obj
            objs_one_video.append(objs_one_frame)
        vid.objs = objs_one_video
        vid.dir = vid_dir
        vid.id = vid_id
        vid.n_frame = vid_n_frame
        vidb.append(vid)
        print(datetime.now(), 'Finish video %d' %vid_id)
    print('Starting pickle the vidb into file')
    pickle.dump(vidb, open(vidb_f, 'wb'))

    return object_num, frame_num
if __name__ == '__main__':

    t_start = time.time()
    object_num_t, frame_num_t = generate_vidb(config.anno_path_t, config.vid_info_t, config.vidb_t)
    t_end = time.time()
    print('The time for generating training imdb is %f seconds:' %(t_end - t_start))

    t_start = time.time()
    object_num_v, frame_num_v = generate_vidb(config.anno_path_v, config.vid_info_v, config.vidb_v)
    t_end = time.time()
    print('The time for generating validation imdb is %f seconds:' %(t_end - t_start))

    f = open('VID_Info/vid_summary.txt', 'w')
    f.write('Train\n object_num: %d frame_num: %d\n\nVal\n object_num: %d frame_num: %d'
            % (object_num_t, frame_num_t, object_num_v, frame_num_v))
