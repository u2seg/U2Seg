# %%
import os
os.environ["USL_MODE"] = "USLT"

import torchvision.models as models
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from utils import cfg, logger, print_b

utils.init(default_config_file="configs/ImageNet_usl-t_0.2.yaml")

logger.info(cfg)

# %%
print_b("Loading dataset")
assert cfg.DATASET.NAME in [
    "imagenet", "imagenet100"], f"{cfg.DATASET.NAME} is not imagenet or imagenet100"
imagenet100 = cfg.DATASET.NAME == "imagenet100"
num_classes = 100 if imagenet100 else 1000

train_memory_dataset, train_memory_loader = utils.train_memory_imagenet(
    transform_name=cfg.DATASET.TRANSFORM_NAME,
    batch_size=cfg.DATALOADER.BATCH_SIZE,
    workers=cfg.DATALOADER.WORKERS)

targets = torch.tensor(train_memory_dataset.targets)
targets.shape

# %%
print_b("Loading model")

checkpoint = torch.load(cfg.MODEL.PRETRAIN_PATH, map_location="cpu")

num_heads = 3
num_labeled = cfg.USLT.NUM_CLUSTERS
final_sample_num = cfg.USLT.NUM_SELECTED_SAMPLES

backbone = models.__dict__[cfg.MODEL.ARCH](pretrained=False)
backbone.fc = nn.Identity()
# BACKBONE_DIM is used with ContrastiveModel
backbone = utils.ContrastiveModel(
    backbone, head='mlp', features_dim=128, backbone_dim=cfg.MODEL.BACKBONE_DIM)
# Apply L2 Norm with 128 dim feat (not BACKBONE_DIM as in smaller datasets)
model = utils.ClusteringModel(backbone, nclusters=num_labeled, normed=True,
                              nheads=num_heads, backbone_dim=128).cuda()

model.load_state_dict(utils.single_model(checkpoint["model"]))
model.eval()

# %%
print_b("Loading feats list")
feats_list = utils.get_feats_list(
    backbone, train_memory_loader, dataparallel=False, recompute=cfg.RECOMPUTE_ALL, force_no_extra_kwargs=True)

assignment_lists = []
num_parts = 12
feats_parts = feats_list.chunk(num_parts)
cluster_heads = model.cluster_head
for cluster_head in cluster_heads:
    assignment_list = torch.cat(
        [cluster_head(feat_part.cuda()).cpu() for feat_part in tqdm(feats_parts)], dim=0)
    assignment_lists.append(assignment_list)

# %%
all_targets = torch.tensor(train_memory_dataset.targets)

all_preds_list = [assignment_list.argmax(
    dim=1) for assignment_list in assignment_lists]
all_probs_list = [F.softmax(assignment_list, dim=1)
                  for assignment_list in assignment_lists]
all_max_probs_list = [all_probs.max(
    dim=1).values for all_probs in all_probs_list]

head_in_use = cfg.USLT.HEAD_IN_USE
all_max_probs_mean = torch.stack(all_max_probs_list, dim=0).mean(dim=0)
max_probs_in_use = all_max_probs_mean if cfg.USLT.USE_MEAN_PROB else all_max_probs_list[
    head_in_use]

# %%


def get_n_correct(all_preds, all_targets, num_labeled, num_classes):
    n_correct = 0
    for i in range(num_labeled):
        labeled_i_mask = all_preds == i
        labeled_i_target = torch.bincount(
            all_targets[labeled_i_mask], minlength=num_classes)
        # print(labeled_i_target)
        # print(labeled_i_target.max().item())
        n_correct += torch.max(labeled_i_target)  # Best Match
    return n_correct

# See the GT label distribution of selected_inds


def get_dist(selected_inds, num_classes):
    selected_inds_label = []

    for ind in selected_inds:
        selected_inds_label.append(all_targets[ind])

    return np.bincount(selected_inds_label, minlength=num_classes)

# %%


recompute_num_dependent = cfg.RECOMPUTE_ALL or cfg.RECOMPUTE_NUM_DEP

# %%
print_b("Calculating first order kNN density estimation")
d_knns, ind_knns = utils.partitioned_kNN(
    feats_list, K=cfg.USLT.KNN_K, recompute=cfg.RECOMPUTE_ALL)
neighbors_dist = d_knns.mean(dim=1)

# %%
if imagenet100:
    _, chosen_percent = utils.get_sample_info_imagenet100(
        final_sample_num=final_sample_num)
else:
    _, chosen_percent = utils.get_sample_info_imagenet(
        final_sample_num=final_sample_num)

# %%


def get_n_correct_neighbors_dist(all_preds, all_targets, num_labeled, num_classes, head_id=None):
    n_correct = 0
    for i in range(num_labeled):
        labeled_i_mask = all_preds == i
        labeled_i_targets = all_targets[labeled_i_mask]
        if len(labeled_i_targets) == 0:
            continue
        # Sample that has the minimum distance
        labeled_i_reference_ind = neighbors_dist[labeled_i_mask].argmin()
        # print(labeled_i_reference_ind, neighbors_dist[labeled_i_mask])
        labeled_i_reference_target = labeled_i_targets[labeled_i_reference_ind]

        # print(torch.sum(labeled_i_targets == labeled_i_reference_target).item(), "/", len(labeled_i_targets))
        #
        n_correct += torch.sum(labeled_i_targets == labeled_i_reference_target)
    return n_correct


# %%
# Use all labels (standard evaluate_head)
def get_n_correct_indep(all_preds, all_targets, num_labeled, num_classes, head_id=None):
    n_correct = 0
    for i in range(num_labeled):
        labeled_i_mask = all_preds == i
        labeled_i_target = torch.bincount(
            all_targets[labeled_i_mask], minlength=num_classes)
        # print(labeled_i_target)
        # print(labeled_i_target.max().item())
        n_correct += torch.max(labeled_i_target)  # Best Match
    return n_correct


def evaluate_head(head, get_n_correct):
    n_correct = get_n_correct(
        all_preds_list[head], all_targets, num_labeled, num_classes, head_id=head)

    acc = n_correct.item() / len(all_targets)

    print("Head: {} Acc: {:.2f}".format(head, acc * 100.))


# %%
print("Independent select: optimal matching")
for head in range(num_heads):
    evaluate_head(head, get_n_correct=get_n_correct_indep)

# %%
print("Pick min dist to label")
for head in range(num_heads):
    evaluate_head(head, get_n_correct=get_n_correct_neighbors_dist)


def gen_pseudo_labels_csv_data(save_filename, pseudo_labels, train_memory_dataset, gen_rem=False):
    selected_inds = np.where(pseudo_labels != -1)[0]

    # gen_rem: generate remaining data by excluding the selected data
    if gen_rem:
        rem_set = set(range(len(train_memory_dataset.imgs)))
        rem_set = rem_set - set(list(selected_inds))
        selected_inds = np.array(list(rem_set))
    selected_inds = np.sort(selected_inds)
    print("Number of samples:", len(selected_inds))
    d = []
    for ind in selected_inds:
        # if gen_rem:
        # print(ind)
        d.append([ind, utils.get_img_path_from_full_path(
            train_memory_dataset.imgs[ind]), pseudo_labels[ind]])

    filename = "{}.csv".format(save_filename)
    assert not os.path.exists(filename), "path {} exists".format(filename)
    df = pd.DataFrame(data=d, columns=["Index", "ImageID", "Pseudolabel"])
    df.to_csv(filename, index=False)

def save_pseudo_labels_data(gen_mode, pseudo_labels, ours_filename_part, chosen_percent, train_memory_dataset):
    print("Generation mode:", gen_mode)
    assert gen_mode == "ours"

    filename_part = ours_filename_part

    for gen_rem in [False, True]:
        if gen_rem:
            filename = "train_{}p_gen_{}_index".format(100 - chosen_percent, filename_part)
        else:
            filename = "train_{}p_gen_{}_index".format(chosen_percent, filename_part)
        filename = os.path.join(cfg.RUN_DIR, filename)
        print("Filename:", filename)
        gen_pseudo_labels_csv_data(filename, pseudo_labels, train_memory_dataset, gen_rem=gen_rem)

# Predict only top `top_to_assign_pseudo_label` samples per cluster with pseudo labels, argsort is a little different from argmin result, use `target_num_samples` (2911)
def generate_pseudo_labels_top(all_preds, all_targets, num_labeled, num_classes, head_id, target_num_samples, top_to_assign_pseudo_label = 10):
    pred_counts = np.bincount(all_preds, minlength=num_classes)

    generated_pseudo_labels = torch.tensor([-1] * len(all_targets))
    
    query_count = 0
    for i in tqdm(pred_counts.argsort()[::-1][:target_num_samples]):
        if query_count == target_num_samples:
            break
            
        labeled_i_mask = all_preds == i
        labeled_i_masked_inds = torch.where(labeled_i_mask)[0]
        del labeled_i_mask
        
        if len(labeled_i_masked_inds) == 0:
            continue
        
        query_count += 1
        labeled_i_masked_inds = labeled_i_masked_inds[neighbors_dist[labeled_i_masked_inds].argsort()]
        
        # Sample that has the minimum mean distance
        labeled_i_reference_target = all_targets[labeled_i_masked_inds[0]]
        
        local_ids_to_assign = labeled_i_masked_inds[:top_to_assign_pseudo_label]

        generated_pseudo_labels[local_ids_to_assign] = labeled_i_reference_target
        
    assert query_count == target_num_samples, "{} != {}".format(query_count, target_num_samples)
    print("Generating pseudo labels by requesting labels from {} samples".format(query_count))
    return generated_pseudo_labels

# top 10 or 20
top_to_assign_pseudo_label = cfg.USLT.NUM_PSEUDO_LABELS_PER_CLUSTER
print("Assigning {} labels at maximum from a pseudo label".format(top_to_assign_pseudo_label))

pseudo_labels = []
for head in range(num_heads):
    n_non_empty_clusters = len(np.unique(all_preds_list[head]))
    actual_num_samples = min(final_sample_num, n_non_empty_clusters)
    if n_non_empty_clusters < final_sample_num:
        logger.warning(f"Ask for {final_sample_num} samples but only {n_non_empty_clusters} clusters are non-empty (head {head})")
    pseudo_labels_item = generate_pseudo_labels_top(all_preds_list[head], all_targets, num_labeled, num_classes, target_num_samples=actual_num_samples, head_id=head, top_to_assign_pseudo_label=top_to_assign_pseudo_label)
    pseudo_labels_item_mask = pseudo_labels_item != -1
    pseudo_labels_acc = torch.mean((pseudo_labels_item[pseudo_labels_item_mask] == all_targets[pseudo_labels_item_mask]).float()) * 100.
    print("Head {}, Pseudo-label Acc: {:.2f}".format(head, pseudo_labels_acc))
    pseudo_labels.append(pseudo_labels_item.numpy())

n_non_empty_clusters = len(np.unique(all_preds_list[head_in_use]))
actual_num_samples = min(num_labeled, n_non_empty_clusters)
assert actual_num_samples <= num_labeled

ours_filename_part = cfg.RUN_NAME
save_pseudo_labels_data(gen_mode="ours", pseudo_labels=pseudo_labels[head_in_use],
        ours_filename_part=ours_filename_part, chosen_percent=chosen_percent, train_memory_dataset=train_memory_dataset)
