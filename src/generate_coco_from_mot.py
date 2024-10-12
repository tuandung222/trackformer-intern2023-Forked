# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Generates COCO data and annotation structure from MOTChallenge data.
"""
import argparse
import configparser
import csv
import json
import os
import shutil

import numpy as np
# import pycocotools.mask as rletools
from skimage import io
import torch
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
from scipy.optimize import linear_sum_assignment
from torchvision.ops.boxes import box_iou

from trackformer.datasets.tracking.mots20_sequence import load_mots_gt


VIS_THRESHOLD = 0.25
DATA_ROOT = 'data/BKTRIS_TRAINING/'
SEQ_TRAINS = [f'seq_{i}_train' for i in range(3)]
SEQ_VALS = [f'seq_{i}_val' for i in range(2)]
SEQS = SEQ_TRAINS + SEQ_VALS




global_categories = GLOBAL_CATEGORIES = [{'id': 1, 'name': 'person-walking', 'supercategory': 'person-only'},
 {'id': 2, 'name': 'motorbike', 'supercategory': 'person-vehicle'},
 {'id': 3, 'name': 'car', 'supercategory': 'vehicle-only'},
 {'id': 4, 'name': 'truck', 'supercategory': 'vehicle-only'}]
GLOBAL_CLASS_ID_DICT = {c['name']: c['id'] for c in GLOBAL_CATEGORIES}
# BKTRIS_INFO = {
#     'seq_0_train': {
#         'img_width': 2560,
#         'img_height': 1440,
#     },
#     'seq_0_val': {
#         'img_width': 2560,
#         'img_height': 1440,
#     }
# }

def generate_coco_from_mot(split_name='train', seqs_names=SEQS,
                           root_split='train', mots=False, mots_vis=False,
                           frame_range=None, data_root=DATA_ROOT, categories=GLOBAL_CATEGORIES):
    """
    Generates COCO data from MOT.
    """
    
    if frame_range is None:
        frame_range = {'start': 0.0, 'end': 1.0}
    #PATH
    root_split_path = os.path.join(data_root, root_split)
    coco_dir = os.path.join(data_root, split_name)
    if os.path.isdir(coco_dir):
        shutil.rmtree(coco_dir)
    os.mkdir(coco_dir)
    
    #ANNOTATION FORMAT
    annotations = {}
    annotations['type'] = 'instances'
    annotations['images'] = []
    annotations['categories'] = global_categories
    annotations['annotations'] = []
    annotations_dir = os.path.join(os.path.join(data_root, 'annotations'))
    if not os.path.isdir(annotations_dir):
        os.mkdir(annotations_dir)
    annotation_file = os.path.join(annotations_dir, f'{split_name}.json')

    # IMAGE FILES
    img_id = 0
    seqs = sorted(os.listdir(root_split_path))
    frame_id_dict = {}

    if seqs_names is not None:
        seqs = [s for s in seqs if s in seqs_names]
    annotations['sequences'] = seqs
    annotations['frame_range'] = frame_range
    print(split_name, seqs)
    annotations['sequences'] = seqs
    annotations['frame_range'] = frame_range

    for seq in seqs:

        if True:
            img_width = 2560
            img_height = 1440
            
            
            
            seq_length = len(os.listdir(os.path.join(root_split_path, seq, 'img1')))

        seg_list_dir = sorted(os.listdir(os.path.join(root_split_path, seq, 'img1')))
        seg_list_dir = [x for x in seg_list_dir if '.jpg' in x or '.png' in x]
        start_frame = int(frame_range['start'] * seq_length)
        end_frame = int(frame_range['end'] * seq_length)
        seg_list_dir = seg_list_dir[start_frame: end_frame]

        print(f"{seq}: {len(seg_list_dir)}/{seq_length}")
        seq_length = len(seg_list_dir)
        
        for i, img in enumerate(sorted(seg_list_dir)):

            if i == 0:
                first_frame_image_id = img_id

            annotations['images'].append({"file_name": f"{seq}_{img}",
                                          "height": img_height,
                                          "width": img_width,
                                          "id": img_id,
                                          "frame_id": i,
                                          "seq_length": seq_length,
                                          "first_frame_image_id": first_frame_image_id})

            img_id += 1

            os.symlink(os.path.join(os.getcwd(), root_split_path, seq, 'img1', img),
                       os.path.join(coco_dir, f"{seq}_{img}"))

    # GT
    annotation_id = 0
    img_file_name_to_id = {img_dict['file_name']: img_dict['id'] for img_dict in annotations['images']}
    for seq in seqs:
        seq_annotations = []
        if True:
            seq_annotations_per_frame = {}
            json_file_path = os.path.join('c_annotations', seq, 'annotations.json')
            with open(json_file_path, "r") as js_file:
                js_objs = json.load(js_file)
                for js_obj in js_objs:
                    # pth = js_obj['path'].split('/')
                    bounding_boxes = js_obj['boundingBoxes']
                    for obj in bounding_boxes:
                        if 'object_id' not in obj:
                            track_id = 9999 + len(bounding_boxes)
                        else:
                            track_id = int(obj['object_id'])
                        top, left, width, height = obj['top'], obj['left'], obj['width'], obj['height']
                        bbox = [left, top, width, height]
                        area = bbox[2] * bbox[3]
                        visibility = 1
                        if seq not in frame_id_dict:
                            frame_id_dict[seq] = len(frame_id_dict) + 1
                        frame_id = frame_id_dict[seq]
                        if 'class_id' in obj:
                            category_id = obj['class_id']
                        else: 
                            category_id = 1
                        
                        image_id = img_file_name_to_id[f'{seq}_{js_obj["frame"]}']
                        # track_id = int(obj['object_id'])

                        annotation = {
                            "id": annotation_id,
                            "bbox": bbox,
                            "image_id": image_id,
                            "segmentation": [],
                            "ignore": 0 if visibility > VIS_THRESHOLD else 1,
                            "visibility": visibility,
                            "area": area,
                            "iscrowd": 0,
                            "seq": seq,
                            "category_id": category_id,
                            "track_id": track_id}

                        seq_annotations.append(annotation)
                        if frame_id not in seq_annotations_per_frame:
                            seq_annotations_per_frame[frame_id] = []
                        seq_annotations_per_frame[frame_id].append(annotation)

                        annotation_id += 1

            annotations['annotations'].extend(seq_annotations)


    # max objs per image
    num_objs_per_image = {}
    for anno in annotations['annotations']:
        image_id = anno["image_id"]
        if image_id in num_objs_per_image:
            num_objs_per_image[image_id] += 1
        else:
            num_objs_per_image[image_id] = 1

    if len(num_objs_per_image) > 0:
        print(f'max objs per image: {max(list(num_objs_per_image.values()))}')

    with open(annotation_file, 'w') as anno_file:
        json.dump(annotations, anno_file, indent=4)


def check_coco_from_mot(coco_dir='data/BKTRIS_TRAINING/train_coco', annotation_file='data/BKTRIS_TRAINING/annotations/train_coco.json', img_id=200):
    """
    Visualize generated COCO data. Only used for debugging.
    """
    # coco_dir = os.path.join(data_root, split)
    # annotation_file = os.path.join(coco_dir, 'annotations.json')

    coco = COCO(annotation_file)
    cat_ids = coco.getCatIds(catNms=['person'])
    if img_id == None:
        img_ids = coco.getImgIds(catIds=cat_ids)
        index = np.random.randint(0, len(img_ids))
        img_id = img_ids[index]
    img = coco.loadImgs(img_id)[0]

    i = io.imread(os.path.join(coco_dir, img['file_name']))

    plt.imshow(i)
    plt.axis('off')
    ann_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    coco.showAnns(anns, draw_bbox=True)
    plt.show()
    plt.savefig('annotations.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate COCO from MOT.')
    parser.add_argument('--mots20', action='store_true')
    parser.add_argument('--mot20', action='store_true')
    args = parser.parse_args()
    check_coco_from_mot()
    generate_coco_from_mot(
        'train_coco',
        seqs_names=SEQ_TRAINS)

    generate_coco_from_mot(
        'val_coco',
        seqs_names=SEQ_VALS, root_split="val")
    check_coco_from_mot()

