#!/usr/bin/env bash
VID_DIR='./VID_Info'
VID_INFO_TRAIN='vid_info_train.txt'
VID_INFO_VAL='vid_info_val.txt'
VIDB_TRAIN='vidb_train.pk'
VIDB_VAL='vidb_val.pk'

if [ ! -d $VID_DIR ]; then
    mkdir $VID_DIR
fi

if [ ! -f $VID_DIR/$VID_INFO_TRAIN ] || [ ! -f $VID_DIR/$VID_INFO_VAL ]; then
    echo 'Start collecting video data information.....'
    python3 collect_vid_info.py
fi

if [ ! -f $VID_DIR/VIDB_TRAIN ] || [ ! -f $VID_DIR/$VIDB_VAL ]; then
    echo 'Start generating video data base.....'
    python3 generate_vidb.py
fi

echo 'Start building tfrecords.....'
python3 build_data_vid.py