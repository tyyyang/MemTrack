# ------------------------------------------------------------------
# Tensorflow implementation of
#  "Learning Dynamic Memory Networks for Object Tracking", ECCV,2018
# Licensed under The MIT License [see LICENSE for details]
# Written by Tianyu Yang (tianyu-yang.com)
# ------------------------------------------------------------------

import os
import sys
sys.path.append('../')
import config
import time


def collect_video_info(data_path, save_path, is_train):
    dirs1 = sorted(os.listdir(data_path))

    video_id = 0
    f_vid_info = open(save_path, 'w')
    if is_train:
        for dir1 in dirs1:
            dirs2 = sorted(os.listdir(os.path.join(data_path, dir1)))
            for dir2 in dirs2:
                files = os.listdir(os.path.join(data_path, dir1, dir2))
                video_dir = os.path.join(dir1, dir2)
                video_id += 1
                n_frames = len(files)
                f_vid_info.write('%s %d %d\n' %(video_dir, video_id, n_frames))
    else:
        for dir1 in dirs1:
            files = os.listdir(os.path.join(data_path, dir1))
            video_id += 1
            n_frames = len(files)
            f_vid_info.write('%s %d %d\n' %(dir1, video_id, n_frames))

    f_vid_info.close()


if __name__=='__main__':
    t_start = time.time()
    collect_video_info(config.data_path_t, config.vid_info_t, True)
    t_end = time.time()
    print('The time for collecting training information is %f seconds:' % (t_end - t_start))

    t_start = time.time()
    collect_video_info(config.data_path_v, config.vid_info_v, False)
    t_end = time.time()
    print('The time for collecting training information is %f seconds:' % (t_end - t_start))