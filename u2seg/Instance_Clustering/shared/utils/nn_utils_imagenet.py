# Datasets
import os
import random
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
from pykeops.torch import LazyTensor
from tqdm import tqdm
import pandas as pd
import pdb

from .config_utils import cfg
from .nn_utils import get_transform, normalization_kwargs_dict
from .uslt_utils import LocalGlobalDataset

def get_clip_model(name):
    import clip
    model, preprocess = clip.load(name)
    model.eval()
    model.cuda()

    return model, preprocess

def get_img_path_from_full_path(full_path):
    img_path = full_path[0]
    sep = "/"
    img_path = sep.join(img_path.split(sep)[-2:])
    return img_path

def gen_csv_data(save_filename, selected_inds, train_memory_dataset, gen_rem=False):
    if isinstance(selected_inds, torch.Tensor):
        print("***Please use numpy array as selected_inds***")
    # gen_rem: generate remaining data by excluding the selected data
    if gen_rem:
        rem_set = set(range(len(train_memory_dataset.imgs)))
        rem_set = rem_set - set(list(selected_inds))
        selected_inds = np.array(list(rem_set))
    selected_inds = np.sort(selected_inds)
    print(len(selected_inds))
    d = []
    for ind in selected_inds:
        d.append([ind, get_img_path_from_full_path(train_memory_dataset.imgs[ind])])

    filename = "{}.csv".format(save_filename)
    assert not os.path.exists(filename), "path {} exists".format(filename)
    df = pd.DataFrame(data=d, columns=["Index", "ImageID"])
    df.to_csv(filename, index=False)

def save_data(gen_mode, stratified_density_selected_data_output, ours_filename_part, feats_list, final_sample_num, chosen_percent, train_memory_dataset):
    print("Generation mode:", gen_mode)

    if gen_mode == "ours":
        selected_inds = stratified_density_selected_data_output
        filename_part = ours_filename_part
    elif gen_mode == "random":
        np.random.seed(0)
        selected_inds = np.random.choice(feats_list.size(0), size=(final_sample_num,), replace=False)
        filename_part = "random"
    else:
        raise ValueError("gen_mode: " + gen_mode)

    for gen_rem in [False, True]:
        if gen_rem:
            filename = "train_{}p_gen_{}_index".format(100 - chosen_percent, filename_part)
        else:
            filename = "train_{}p_gen_{}_index".format(chosen_percent, filename_part)
        filename = os.path.join(cfg.RUN_DIR, filename)
        print("Filename:", filename)
        gen_csv_data(filename, selected_inds, train_memory_dataset, gen_rem=gen_rem)


def get_sample_info_imagenet100(final_sample_num):
    if final_sample_num == 400:
        # 0.3 percent
        num_centroids = 400
        chosen_percent = 0.3
    else:
        raise ValueError(final_sample_num)

    return num_centroids, chosen_percent

def get_sample_info_imagenet(final_sample_num):
    if final_sample_num == 12820:
        # 1 percent
        num_centroids = 12900
        chosen_percent = 1
    elif final_sample_num == 2911:
        # 0.2 percent
        num_centroids = 2911
        chosen_percent = 0.2
    else:
        # raise ValueError(final_sample_num)
        num_centroids = final_sample_num 
        chosen_percent = 0.2 * final_sample_num / 2911

    return num_centroids, chosen_percent


def get_selection_with_reg_imagenet(data, neighbors_dist, cluster_labels, num_centroids,
                  iters=1, final_sample_num=None, w=1, momentum=0.5, horizon_num=256, alpha=1, exclude_same_cluster=False, verbose=False):
    # Intuition: horizon_num = dimension * 2

    cluster_labels_cuda = cluster_labels.cuda()
    neighbors_dist_cuda = neighbors_dist.cuda()
    selection_regularizer = torch.zeros_like(neighbors_dist_cuda)

    data = data.cuda()
    N, D = data.shape  # Number of samples, dimension of the ambient space

    data_expanded_lazy = LazyTensor(data.view(N, 1, D))

    for iter_ind in tqdm(range(iters)):
        selected_inds = []
        if verbose:
            print("Computing selected ids")
            print("selection_regularizer", selection_regularizer)
        for cls_ind in range(num_centroids):
            if len(selected_inds) == final_sample_num:
                break
            match_arr = cluster_labels_cuda == cls_ind
            match = torch.where(match_arr)[0]
            if len(match) == 0:
                continue

            # scores in the selection process

            # No prior:
            # scores = 1 / neighbors_dist[match_arr]

            scores = 1 / \
                neighbors_dist_cuda[match_arr] - w * \
                selection_regularizer[match_arr]
            if iter_ind != 0 and cls_ind == 0 and verbose:
                print("original score:", (1 / neighbors_dist_cuda[match_arr]).mean(),
                      "regularizer adjustment:", (w * selection_regularizer[match_arr]).mean())
            min_dist_ind = scores.argmax()
            selected_inds.append(match[min_dist_ind].item())

        selected_inds = torch.tensor(selected_inds)

        if iter_ind < iters - 1:  # Not the last iteration
            if verbose:
                print("Updating selection regularizer")

            selected_data = data[selected_inds]

            if not exclude_same_cluster:
                # This is square distances: (N_full, N_selected)
                # data: (N_full, 1, dim)
                # selected_data: (1, N_selected, dim)
                new_selection_regularizer = (
                    (data_expanded_lazy - selected_data[None, :, :]) ** 2).sum(dim=-1)
                new_selection_regularizer = new_selection_regularizer.Kmin(
                    horizon_num, dim=1)

                if verbose:
                    print("new_selection_regularizer shape:",
                        new_selection_regularizer.shape)
                    print("Max:", new_selection_regularizer.max())
                    print("Mean:", new_selection_regularizer.mean())

                # Distance to oneself should be ignored
                new_selection_regularizer[new_selection_regularizer == 0] = 1e10
            else:
                # This is square distances: (N_full, N_selected)
                # data: (N_full, 1, dim)
                # selected_data: (1, N_selected, dim)

                # We take the horizon_num samples with the min distance to the other centroids
                new_selection_regularizer = (
                    (data_expanded_lazy - selected_data[None, :, :]) ** 2).sum(dim=-1)
                # indices within selected data
                new_selection_regularizer, selected_data_ind = new_selection_regularizer.Kmin_argKmin(horizon_num,
                                                                                                      dim=1, backend="GPU")

                if verbose:
                    print("new_selection_regularizer shape:",
                        new_selection_regularizer.shape)
                    print("Max:", new_selection_regularizer.max())
                    print("Mean:", new_selection_regularizer.mean())

                # Distance to the instance in the same cluster should be ignored (including oneself if the sample is currently selected)
                # **NOTE**: if some clusters are skipped, select_data_ind may not match cluster_labels
                # This does not happen in 0.2% case, but could happen in 1% case.
                same_cluster_selected_data_ind_mask = (
                    selected_data_ind == cluster_labels_cuda.view((-1, 1))).float()
                # It's true that if cluster is not in the list, some instances will have one more regularizer item, but this is a small contribution.
                new_selection_regularizer = (1 - same_cluster_selected_data_ind_mask) * \
                    new_selection_regularizer + same_cluster_selected_data_ind_mask * 1e10

                assert not torch.any(new_selection_regularizer == 0), "{}".format(
                    torch.where(new_selection_regularizer == 0))

            if verbose:
                print("Min:", new_selection_regularizer.min())

            # selection_regularizer: N_full
            if alpha != 1:
                new_selection_regularizer = (
                    1 / new_selection_regularizer ** alpha).sum(dim=1)
            else:
                new_selection_regularizer = (
                    1 / new_selection_regularizer).sum(dim=1)

            selection_regularizer = selection_regularizer * \
                momentum + new_selection_regularizer * (1 - momentum)
    del cluster_labels_cuda
    del neighbors_dist_cuda
    del data

    assert len(selected_inds) == final_sample_num
    return selected_inds.numpy()


# Credit: MoCov2 https://github.com/facebookresearch/moco/blob/main/moco/loader.py


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class L2NormLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.normalize(x, dim=1)

# Credit: https://github.com/amazon-research/exponential-moving-average-normalization


def build_hidden_head(num_mlp, dim_mlp, dim, normed=False):
    modules = []
    for _ in range(1, num_mlp):
        modules.append(nn.Linear(dim_mlp, dim_mlp))
        modules.append(nn.ReLU())
    modules.append(nn.Linear(dim_mlp, dim))
    if normed:
        modules.append(L2NormLayer())
    return nn.Sequential(*modules)

# Credit: https://github.com/wvangansbeke/Unsupervised-Classification/blob/master/data/imagenet.py


class ImageNet(datasets.ImageFolder):
    def __init__(self, root, split='train', transform=None):
        super(ImageNet, self).__init__(root=os.path.join(root, split),
                                       transform=None)
        self.transform = transform
        self.split = split
        self.resize = transforms.Resize(256)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        im_size = img.size
        img = self.resize(img)

        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': target, 'meta': {
            'im_size': im_size, 'index': index}}

        return out

    def get_image(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self.resize(img)
        return img


# class ImageNetSubset(torch.utils.data.Dataset):
#     def __init__(self, subset_file, root, split='train',
#                  transform=None, return_dict=True):
#         super(ImageNetSubset, self).__init__()

#         self.root = os.path.join(root, split)
#         self.transform = transform
#         self.split = split

#         # Read the subset of classes to include (sorted)
#         with open(subset_file, 'r') as f:
#             result = f.read().splitlines()
#         subdirs, class_names = [], []
#         for line in result:
#             subdir, class_name = line.split(' ', 1)
#             subdirs.append(subdir)
#             class_names.append(class_name)

#         # Gather the files (sorted)
#         imgs = []
#         targets = []
#         for i, subdir in enumerate(subdirs):
#             # subdir_path = os.path.join(self.root, subdir)

#             files = sorted(glob(os.path.join(self.root, subdir, '*.JPEG')))
#             for f in files:
#                 imgs.append((f, i))
#                 targets.append(i)

#         self.imgs = imgs
#         self.classes = class_names
#         self.targets = targets
#         self.resize = transforms.Resize(256)

#         self.return_dict = return_dict

#     def get_image(self, index):
#         path, target = self.imgs[index]
#         with open(path, 'rb') as f:
#             img = Image.open(f).convert('RGB')
#         img = self.resize(img)
#         return img

#     def __len__(self):
#         return len(self.imgs)

#     def __getitem__(self, index):
#         path, target = self.imgs[index]
#         with open(path, 'rb') as f:
#             img = Image.open(f).convert('RGB')
#         im_size = img.size
#         if self.return_dict:
#             img = self.resize(img)
#         class_name = self.classes[target]

#         if self.transform is not None:
#             img = self.transform(img)

#         if self.return_dict:
#             out = {'image': img, 'target': target, 'meta': {
#                 'im_size': im_size, 'index': index, 'class_name': class_name}}

#             return out
#         return img, target


def train_dataset_imagenet(transform_name, load_local_global_dataset=False, add_memory_bank_dataset=False):
    # Uses MoCov2 aug: https://github.com/facebookresearch/moco/blob/main/main_moco.py
    if transform_name == "imagenet" or transform_name == "imagenet100":
        normalization_kwargs = normalization_kwargs_dict[transform_name]
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**normalization_kwargs)])

        # val uses the original dataloader (which doesn't add additional Resize transform)
        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(**normalization_kwargs)])

        train_transforms = {
            'standard': transform_val, 'augment': transform_train}
    else:
        raise ValueError(f"Unsupported transform type: {transform_name}")

    if load_local_global_dataset:
        # Transform is set on the wrapper if load_local_global_dataset is True
        transform_train = None

    train_dataset = ImageNet(
        root=cfg.DATASET.ROOT_DIR, split='train', transform=transform_train)

    val_dataset = datasets.ImageFolder(
        root=os.path.join(cfg.DATASET.ROOT_DIR, 'val'), transform=transform_val)

    if load_local_global_dataset:
        indices = np.load(cfg.USLT_PRETRAIN.TOPK_NEIGHBORS_PATH)
        train_dataset = LocalGlobalDataset(
            dataset=train_dataset, neighbors_indices=indices, num_neighbors=cfg.USLT_PRETRAIN.NUM_NEIGHBORS, transform=train_transforms)

    if add_memory_bank_dataset:
        # memory_bank_dataset: training dataset with validation data augmentation
        memory_bank_dataset = datasets.ImageFolder(
            root=os.path.join(cfg.DATASET.ROOT_DIR, 'train'), transform=transform_val)
        return train_dataset, val_dataset, memory_bank_dataset
    
    return train_dataset, val_dataset


# Memory bank on ImageNet
def train_memory_imagenet(transform_name, batch_size=128, workers=2, with_val=False, transform_override=None):
    if transform_override is None:
        transform_test = get_transform(transform_name)
    else:
        transform_test = transform_override

    assert cfg.DATASET.NAME == "imagenet" or cfg.DATASET.NAME == "imagenet100", f"Dataset should be ImageNet or its subset, but get {cfg.DATASET.NAME}"
    train_memory_dataset = datasets.ImageFolder(root=os.path.join(cfg.DATASET.ROOT_DIR, 'train'),
                                                transform=transform_test)
    if with_val:
        val_memory_dataset = datasets.ImageFolder(root=os.path.join(cfg.DATASET.ROOT_DIR, 'val'),
                                                  transform=transform_test)

    train_memory_loader = torch.utils.data.DataLoader(
        train_memory_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, drop_last=False)

    if with_val:
        val_memory_loader = torch.utils.data.DataLoader(
            val_memory_dataset, batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True, drop_last=False)
        return train_memory_dataset, train_memory_loader, val_memory_dataset, val_memory_loader
    else:
        return train_memory_dataset, train_memory_loader
