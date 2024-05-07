#%%
import os

os.environ["USL_MODE"] = "USLT"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.resnet_cifar as resnet_cifar
import utils
from utils import cfg, logger, print_b

utils.init(default_config_file="configs/cifar10_usl-t.yaml")

logger.info(cfg)

#%%
print_b("Loading dataset")
assert cfg.DATASET.NAME in ["cifar10", "cifar100"], f"{cfg.DATASET.NAME} is not cifar10 or cifar100"
cifar100 = cfg.DATASET.NAME == "cifar100"
num_classes = 100 if cifar100 else 10

train_memory_dataset, train_memory_loader = utils.train_memory_cifar(
    root_dir=cfg.DATASET.ROOT_DIR, 
    batch_size=cfg.DATALOADER.BATCH_SIZE, 
    workers=cfg.DATALOADER.WORKERS, transform_name=cfg.DATASET.TRANSFORM_NAME, cifar100=cifar100)

targets = torch.tensor(train_memory_dataset.targets)
targets.shape

#%%
class ClusteringModel(nn.Module):
    def __init__(self, backbone, nclusters, backbone_dim=512, nheads=1, **kwargs):
        super(ClusteringModel, self).__init__()
        self.backbone = backbone
        self.backbone_dim = backbone_dim
        self.nheads = nheads
        assert(isinstance(self.nheads, int))
        assert(self.nheads > 0)
        
        self.cluster_head = nn.ModuleList([nn.Linear(self.backbone_dim, nclusters, **kwargs) for _ in range(self.nheads)])

    def forward(self, x):
        features = self.backbone(x)
        out = [cluster_head(features) for cluster_head in self.cluster_head]
        
        return out

#%%
print_b("Loading model")

checkpoint = torch.load(cfg.MODEL.PRETRAIN_PATH, map_location="cpu")

nheads = 3
num_labeled = cfg.USLT.NUM_SELECTED_SAMPLES

backbone = resnet_cifar.__dict__[cfg.MODEL.ARCH]()
model = ClusteringModel(backbone, nclusters=num_labeled, nheads=nheads, backbone_dim=cfg.MODEL.BACKBONE_DIM).cuda()
model.load_state_dict(utils.single_model(checkpoint["model"]))
model.eval();

#%%
print_b("Loading feat list")
feats_list = utils.get_feats_list(backbone, train_memory_loader, feat_dim=512, recompute=cfg.RECOMPUTE_ALL, force_no_extra_kwargs=True)

#%%
assignment_lists = []
for cluster_head in model.cluster_head:
    assignment_lists.append(cluster_head(feats_list.cuda()).cpu())

#%%
all_targets = torch.tensor(train_memory_dataset.targets)

all_preds_list = [assignment_list.argmax(dim=1) for assignment_list in assignment_lists]
all_probs_list = [F.softmax(assignment_list, dim=1) for assignment_list in assignment_lists]
all_max_probs_list = [all_probs.max(dim=1).values for all_probs in all_probs_list]

head_in_use = cfg.USLT.HEAD_IN_USE
all_max_probs_mean = torch.stack(all_max_probs_list, dim=0).mean(dim=0)
max_probs_in_use = all_max_probs_mean if cfg.USLT.USE_MEAN_PROB else all_max_probs_list[head_in_use]

#%%
def get_n_correct(all_preds, all_targets, num_labeled, num_classes):
    n_correct = 0
    for i in range(num_labeled):
        labeled_i_mask = all_preds == i
        labeled_i_target = torch.bincount(all_targets[labeled_i_mask], minlength=num_classes)
        # print(labeled_i_target)
        # print(labeled_i_target.max().item())
        n_correct += torch.max(labeled_i_target) # Best Match
    return n_correct

# See the GT label distribution of selected_inds
def get_dist(selected_inds, num_classes):
    selected_inds_label = []

    for ind in selected_inds:
        selected_inds_label.append(all_targets[ind])

    return np.bincount(selected_inds_label, minlength=num_classes)

#%%
def get_selection_fn(final_sample_num):
    selected_inds = []
    # Max prob stratified (head `head_in_use`)
    for i in range(final_sample_num):
        labeled_i_mask = all_preds_list[head_in_use] == i
        index_within_cluster = max_probs_in_use[labeled_i_mask].argmax()
        indices = torch.where(labeled_i_mask)[0]
        
        current_select = indices[index_within_cluster]
        current_select = current_select.item()
        selected_inds.append(current_select)

    return selected_inds


recompute_num_dependent = cfg.RECOMPUTE_ALL or cfg.RECOMPUTE_NUM_DEP

#%%
# n_correct = get_n_correct(all_preds_list[head_in_use], all_targets, num_labeled, num_classes)
# acc = n_correct.item() / len(all_preds_list[head_in_use])

# print("Hungarian Matching Acc: {:.2f}".format(acc * 100.))

# `seed` is used as a suffix only
suffix = str(head_in_use)
selected_inds = utils.get_selection(get_selection_fn, final_sample_num=num_labeled, recompute=recompute_num_dependent, save=True, seed=suffix)

print("Selected IDs:")
print(repr(selected_inds))

dist = get_dist(selected_inds, num_classes)

print(repr(dist), dist.std(), (dist > 0).sum())
