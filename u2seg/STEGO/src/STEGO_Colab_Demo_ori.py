#!/usr/bin/env python
# coding: utf-8

import os
from os.path import join
import torch
import torch.nn.functional as F
from PIL import Image
from train_segmentation import LitUnsupervisedSegmenter
from crf import dense_crf
import matplotlib.pyplot as plt
from utils import get_transform, unnorm, remove_axes

# Define directories
saved_models_dir = join("..", "saved_models")
input_images_directory = "/home/xinyang/Panoptic-CutLER/data/trail/resized"
debug_directory = "/home/xinyang/Panoptic-CutLER/data/debug"
os.makedirs(debug_directory, exist_ok=True)

# Load pretrained STEGO
model = LitUnsupervisedSegmenter.load_from_checkpoint(join(saved_models_dir, "cocostuff27_vit_base_5.ckpt")).cuda()

# Process and visualize each image in the directory
for img_name in os.listdir(input_images_directory):
    img_path = os.path.join(input_images_directory, img_name)
    img = Image.open(img_path)
    
    transform = get_transform(448, False, "center")
    img_tensor = transform(img).unsqueeze(0).cuda()

    # Query model and pass result through CRF
    with torch.no_grad():
        code1 = model(img_tensor)
        code2 = model(img_tensor.flip(dims=[3]))
        code = (code1 + code2.flip(dims=[3])) / 2
        code = F.interpolate(code, img_tensor.shape[-2:], mode='bilinear', align_corners=False)
        linear_probs = torch.log_softmax(model.linear_probe(code), dim=1).cpu()
        cluster_probs = model.cluster_probe(code, 2, log_probs=True).cpu()

        single_img = img_tensor[0].cpu()
        linear_pred = dense_crf(single_img, linear_probs[0]).argmax(0)
        cluster_pred = dense_crf(single_img, cluster_probs[0]).argmax(0)

    # Visualize Result and Save
    fig, ax = plt.subplots(1, 3, figsize=(5*3, 5))
    ax[0].imshow(unnorm(img_tensor)[0].permute(1, 2, 0).cpu())
    ax[0].set_title("Image")
    ax[1].imshow(model.label_cmap[cluster_pred])
    ax[1].set_title("Cluster Predictions")
    ax[2].imshow(model.label_cmap[linear_pred])
    ax[2].set_title("Linear Probe Predictions")
    remove_axes(ax)

    output_path = os.path.join(debug_directory, img_name.split('.')[0] + '.png')
    fig.savefig(output_path)
    plt.close(fig)  # Close the plot to free memory