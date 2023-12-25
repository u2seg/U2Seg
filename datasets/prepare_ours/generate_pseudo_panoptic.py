import os
import json
from fvcore.common.download import download
from panopticapi.utils import rgb2id,id2rgb
from PIL import Image
import numpy as np
import pycocotools.mask as mask_util
from tqdm import tqdm
from skimage.measure import label as sk_label
from skimage.measure import regionprops as sk_regions
import argparse

def create_cate(num):
    cate = []
    for i in range(num + 27):
        curr = {}
        curr['supercategory'] = str(i + 1)
        curr['id'] = i + 1
        curr['name'] = str(i + 1)
        if i + 1 <= num:
            curr['isthing'] = 1
        else:
            curr['isthing'] = 0
        cate.append(curr)
    return cate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help')

    parser.add_argument('-class_num', '--class_num', type=int, default=800)  # option that takes a value
    parser.add_argument('-split', '--split',type = str, default='train')  # on/off flag
    args = parser.parse_args()
    class_num = args.class_num
    split = args.split
    template = json.load(open('datasets/datasets/panoptic_anns/panoptic_{}2017.json'.format(split)))
    template_instance = json.load(open('datasets/datasets/coco/annotations/instances_{}2017.json'.format(split)))

    our_ins_pred = json.load(open('datasets/prepare_ours/u2seg_annotations/ins_annotations/coco{}_{}_ins_panoptic.json'.format(split, str(class_num))))

    new_json = {}
    new_json['images'] = template['images']
    new_json['info'] = template['info']
    new_json['licenses'] = template['licenses']
    new_combined_ann = []
    new_json['annotations'] = new_combined_ann

    new_json['categories'] = create_cate(num = class_num)

    # load the semantic dict
    semantic_dict = {}
    iddx = 0
    for line in open("datasets/prepare_ours/u2seg_annotations/semantic_annotations/coco_{}_img_file_names.txt".format(split)):
        s = 1
        semantic_dict[line[:-4] + 'png'] = '{}.npy'.format(str(iddx))
        iddx += 1

    seg_idx = 1

    # img_exist = []
    img_exist = {}
    for _img in tqdm(template['images']):
        img_exist[_img['id']] = False
    for ann in tqdm(template['annotations']):
        current_combine_ann = {}

        file_name = ann['file_name']
        gt_seg_info = ann['segments_info']

        current_combine_ann['file_name'] = file_name
        current_combine_ann['image_id'] = ann['image_id']
        current_combine_ann['segments_info'] = []

        # gt_img = np.asarray(Image.open(os.path.join('/home/niudt/detectron2/datasets/datasets/coco/panoptic_val2017', file_name)), dtype=np.uint32)
        # gt_panoptic = rgb2id(gt_img)

        # load stego inference results
        stego_inf = np.load(os.path.join('datasets/prepare_ours/u2seg_annotations/semantic_annotations/stego_coco_{}_semantic_seg_resized'.format(split), semantic_dict[file_name]))
        stego_inf = stego_inf + 1
        combined_mask = np.zeros(stego_inf.shape, dtype=np.uint32)


        # load pseudo ins annotations
        try:
            pseudo_ins = our_ins_pred['annotations'][str(ann['image_id'])]['segments_info']
        except:
            continue

        # order the instances based on decending area
        area = []
        masks = []
        for ins in pseudo_ins:
            _area = ins['bbox'][-2] * ins['bbox'][-1]
            area.append(_area)
            rle = ins['segmentation']
            mask = mask_util.decode(rle)
            masks.append(mask)
        s = 1

        sorted_id = sorted(range(len(area)), key=lambda k: area[k], reverse=True)
        for ins_idx in sorted_id:
            current_mask = masks[ins_idx]
            current_ins_ann = pseudo_ins[ins_idx]
            s = 1
            combined_mask[current_mask == 1] = seg_idx
            current_ins_ann['id'] = seg_idx
            current_combine_ann['segments_info'].append(current_ins_ann)
            seg_idx += 1
        # delete the all overlapping one
        correct_ins_info = []
        for _ins in current_combine_ann['segments_info']:
            _seg_idx = _ins['id']
            if np.any(combined_mask == _seg_idx):
                correct_ins_info.append(_ins)
            else:
                continue
        current_combine_ann['segments_info'] = correct_ins_info


        # update the semantic part
        for _semantic_cate_id in range(1, 28):
            current_mask = (stego_inf == _semantic_cate_id) * (combined_mask == 0)
            if not np.any(current_mask):
                continue
            overalpping_rate = np.sum((stego_inf == _semantic_cate_id) * (combined_mask != 0))/np.sum(stego_inf == _semantic_cate_id)

            if overalpping_rate > 0.7:
                continue
            combined_mask[current_mask] = seg_idx
            # create an ins_info in segments_info
            semantic_ins = {}
            semantic_ins['category_id'] = _semantic_cate_id + class_num
            semantic_ins['id'] = seg_idx
            semantic_ins['iscrowd'] = 0
            semantic_ins['bbox'] = []
            # sk_mask = sk_label(current_mask)
            # regions = sk_regions(sk_mask)
            # top, left, bottom, right = regions.bbox
            semantic_ins['area'] = 0
            current_combine_ann['segments_info'].append(semantic_ins)
            seg_idx += 1
        # if not ann['image_id'] in img_exist:
        img_exist[ann['image_id']] = True

        combined_mask_panoptic = id2rgb(combined_mask)
        # save the combined_mask
        save_root = 'datasets/prepare_ours/u2seg_annotations/panoptic_annotations/coco{}_{}'.format(split, str(class_num))#'/home/niudt/detectron2/datasets/prepare_ours/uni-training-ann/panoptic_annotations/cocoval_300'
        os.makedirs(save_root, exist_ok=True)
        save_path = os.path.join(save_root, file_name)
        combined_mask_panoptic = Image.fromarray(combined_mask_panoptic)
        combined_mask_panoptic.save(save_path)
        new_combined_ann.append(current_combine_ann)

        # this is test code
        # re_img = np.asarray(
        #     Image.open(save_path),
        #     dtype=np.uint32)
        # re_panoptic = rgb2id(re_img)
        s = 1

    new_img_list = []
    for img in tqdm(new_json['images']):
        if img_exist[img['id']]:
            new_img_list.append(img)

    new_json['images'] = new_img_list
    # save the json
    # json_save_path = os.path.join('/home/niudt/detectron2/datasets/prepare_ours/uni-training-ann/panoptic_annotations/cocoval_300.json')
    json_save_path = os.path.join(
        'datasets/prepare_ours/u2seg_annotations/panoptic_annotations/coco{}_{}.json'.format(split, str(class_num)))
    with open(json_save_path, 'w', encoding='utf-8') as f:
        json.dump(new_json, f, ensure_ascii=False)











