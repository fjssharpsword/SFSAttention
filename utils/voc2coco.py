import os
import argparse
import json
import xml.etree.ElementTree as ET
from typing import Dict, List
from tqdm import tqdm
import re
import torch

# COCO： [x1,y1,w,h] 
def _get_box( points):
    min_x = points[0]
    min_y = points[1]
    max_x = points[2]
    max_y = points[3]
    return [min_x, min_y, max_x, max_y]

# compute area
def _get_area(points):
    min_x = points[0]
    min_y = points[1]
    max_x = points[2]
    max_y = points[3]
    return (max_x - min_x+1) * (max_y - min_y+1)


def voc2coco_target(voc_targets): 
    #input:targets of voc2012, list
    #output: targets of mscoco, list
    VOC_CLASSES = ['background','aeroplane', 'bicycle', 'bird', 'boat','bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person','pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    coco_targets = []
    for i in range(len(voc_targets)):
        
        voc_target = voc_targets[i]['annotation']
        coco_target = {}
        fileid = os.path.splitext(voc_target['filename'])[0]#get ride of extension
        fileid = int("".join(list(filter(str.isdigit, fileid))))
        coco_target["image_id"] = torch.as_tensor(fileid, dtype=torch.int64)
        nums = len(voc_target['object'])
        coco_target["iscrowd"] = torch.zeros((nums,), dtype=torch.int64)
        boxes, labels = [],  []
        width = int(voc_target['size']['width'])
        height = int(voc_target['size']['height'])
        x_scale = 224/width
        y_scale = 224/height
        for j in range(nums):
            obj = voc_target['object'][j]
            points = obj['bndbox']
            points = [int(points['xmin'])*x_scale, int(points['ymin'])*y_scale, \
                      int(points['xmax'])*x_scale, int(points['ymax'])*y_scale] #resize box coordinates to (224,224)
            boxes.append(_get_box(points))
            lbl = obj['name']
            labels.append(VOC_CLASSES.index(lbl))

        coco_target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        coco_target["labels"] = torch.as_tensor(labels, dtype=torch.int64) 
        coco_targets.append(coco_target)
    return coco_targets

def read_coco_classname():
    d = {}
    with open("/data/pycode/SFSAttention/dsts/ms_coco_classnames.txt") as f:
        for line in f:
            (key, val) = line.split(':')
            d[int(key)] = val.replace("\n", "").strip()
    return d

if __name__ == "__main__":
    d = read_coco_classname()
    print(d)