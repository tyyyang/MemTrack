# ------------------------------------------------------------------
# Tensorflow implementation of
#  "Learning Dynamic Memory Networks for Object Tracking", ECCV,2018
# Licensed under The MIT License [see LICENSE for details]
# Written by Tianyu Yang (tianyu-yang.com)
# ------------------------------------------------------------------

import xml.etree.ElementTree as ET
import sys
import os


class BoundingBox(object):
    pass

def get_item(name, root, index=0):
    count = 0
    for item in root.iter(name):
        if count == index:
            return item.text
        count += 1
    # Failed to find "index" occurrence of item.
    return -1


def get_int(name, root, index=0):
    return int(get_item(name, root, index))


def find_num_bb(root):
    index = 0
    while True:
        if get_int('xmin', root, index) == -1:
            break
        index += 1
    return index

def process_xml(xml_file):
    """Process a single XML file containing a bounding box."""
    try:
        tree = ET.parse(xml_file)
    except Exception:
        print('Failed to parse: ' + xml_file, file=sys.stderr)
        return None

    root = tree.getroot()
    num_boxes = find_num_bb(root)
    boxes = []
    for index in range(num_boxes):
        box = BoundingBox()
        # Grab the 'index' annotation.
        box.xmin = get_int('xmin', root, index)
        box.ymin = get_int('ymin', root, index)
        box.xmax = get_int('xmax', root, index)
        box.ymax = get_int('ymax', root, index)
        box.trackid = get_int('trackid', root, index)

        file_name = get_item('filename', root) + '.JPEG'
        folder = get_item('folder', root)

        box.width = get_int('width', root)
        box.height = get_int('height', root)
        box.img_path = os.path.join(folder, file_name)
        box.label = get_item('name', root)
        boxes.append(box)

    return boxes