# this file is to generate annotations with cluster id as category id

# this file is to extract mask from imageget
import os
import shutil
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import json
import cv2
import pycocotools.mask as mask_util
import time
import argparse
import math

def create_category(num):
    cate_list = []
    for i in range(num):
        current_cate = {}
        current_cate['id'] = i + 1
        current_cate['name'] = str(i + 1)
        current_cate['supercategory'] = str(current_cate['id'])
        cate_list.append(current_cate)
    return cate_list






if __name__ == '__main__':
    template = json.load(open('datasets/datasets/coco/annotations/instances_val2017.json'))
    cluster_results = json.load(open('/shared/xinyang/CutLER/UnsupervisedSelectiveLabeling_v1/selective_labeling/saved/coco_val_usl_dino_300/coco_val_usl_dino_800_decode.json')) # the clutsering results
    mask_ann_ori = json.load(open('/home/niudt/detectron2/datasets/prepare_ours/cutler_cocoval_instances_idx.json')) # this is the class-agnostic annotations generated from cutler or maskcut

    new_ann = {}
    new_ann['licenses'] = template['licenses']
    new_ann['categories'] = create_category(num = 300)
    new_ann['images'] = template['images']
    new_ann['info'] = template['info']
    new_ann['annotations'] = []

    print('ori annotations = {}'.format(len(mask_ann_ori)))

    img_exist = []
    count = 0
    for ann in tqdm(mask_ann_ori):
        s = 1
        img_id = str(ann['ins_id']) + '.jpg'
        ann['category_id'] = cluster_results[img_id]
        ann['id'] = ann['ins_id']
        new_ann['annotations'].append(ann)
        if not ann['image_id'] in img_exist:
            img_exist.append(ann['image_id'])


    print('total num of annotations = {}'.format(len(new_ann['annotations'])))

    new_img_list = []
    for img in new_ann['images']:
        if not img['id'] in img_exist:
            continue
        else:
            new_img_list.append(img)

    new_ann['images'] = new_img_list
    print('total images: ', len(new_img_list))

    save_root = './uni-training-ann/ins_annotations/'
    os.makedirs(save_root, exist_ok=True)
    save_path = os.path.join(save_root, 'cocoval_300.json')

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(new_ann, f, ensure_ascii=False)

