#!/usr/bin/env python
# coding: utf-8

import os
from os.path import join
import wget
from train_segmentation import LitUnsupervisedSegmenter
from PIL import Image
from utils import get_transform
import torch.nn.functional as F
from crf import dense_crf
import torch
from pathlib import Path
import tqdm
import json
import numpy as np

# Set GPU device and check GPU status
os.environ["CUDA_VISIBLE_DEVICES"] = "8"

# Download Pretrained Model
print(os.getcwd())
saved_models_dir = join("..", "saved_models")
os.makedirs(saved_models_dir, exist_ok=True)

saved_model_url_root = "https://marhamilresearch4.blob.core.windows.net/stego-public/saved_models/"
saved_model_name = "cocostuff27_vit_base_5.ckpt"
if not os.path.exists(join(saved_models_dir, saved_model_name)):
    wget.download(saved_model_url_root + saved_model_name, join(saved_models_dir, saved_model_name))

# Load pretrained STEGO
model = LitUnsupervisedSegmenter.load_from_checkpoint(join(saved_models_dir, saved_model_name)).cuda()

# Process Image function
def process_image(img_path, model):
    img = Image.open(img_path)
    transform = get_transform(448, False, "center")
    img = transform(img).unsqueeze(0).cuda()

    with torch.no_grad():
        code1 = model(img)
        code2 = model(img.flip(dims=[3]))
        code = (code1 + code2.flip(dims=[3])) / 2
        code = F.interpolate(code, img.shape[-2:], mode='bilinear', align_corners=False)
        linear_probs = torch.log_softmax(model.linear_probe(code), dim=1).cpu()
        cluster_probs = model.cluster_probe(code, 2, log_probs=True).cpu()

        single_img = img[0].cpu()
        linear_pred = dense_crf(single_img, linear_probs[0]).argmax(0)
        cluster_pred = dense_crf(single_img, cluster_probs[0]).argmax(0)

    return linear_pred, cluster_pred

input_directory = Path("/shared/xinyang/Coco/train2017")
output_directory = Path("/shared/xinyang/CutLER/STEGO/data/coco_stego_seg")
output_directory.mkdir(parents=True, exist_ok=True)

all_images = [img for img in input_directory.iterdir() if img.suffix in ['.jpg', '.png']]
semantic_mask_dict = {}


for idx, img_path in enumerate(tqdm.tqdm(all_images)):
    linear_pred, cluster_pred = process_image(img_path, model)
    
    # Save semantic masks as numpy arrays and record paths
    linear_mask_path = output_directory / f"linear_mask_{img_path.stem}.npy"
    cluster_mask_path = output_directory / f"cluster_mask_{img_path.stem}.npy"

    np.save(linear_mask_path, linear_pred)
    np.save(cluster_mask_path, cluster_pred)

    semantic_mask_dict[img_path.name] = {
        'linear_mask': str(linear_mask_path),
        'cluster_mask': str(cluster_mask_path)
    }

    # Debugging: Save results after processing 5 images
    if idx == 4:
        with open(output_directory / 'debug_semantic_mask_dict.json', 'w') as f:
            json.dump(semantic_mask_dict, f)
        break

# Save the dictionary to a JSON file
with open(output_directory / 'semantic_mask_dict.json', 'w') as f:
    json.dump(semantic_mask_dict, f)