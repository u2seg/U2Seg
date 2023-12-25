# Datasets
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
from tqdm import tqdm

from .augment import Augment, Cutout
from .config_utils import cfg, logger
from .nn_utils import get_transform, normalization_kwargs_dict

def get_sample_info_cifar(chosen_sample_num):
    num_centroids = chosen_sample_num
    final_sample_num = chosen_sample_num

    # We get one more centroid to take empty clusters into account
    if chosen_sample_num == 2500:
        num_centroids = 2501
        final_sample_num = 2500
        logger.warning("Returning 2501 as the number of centroids")

    return num_centroids, final_sample_num


def get_selection_with_reg(data, neighbors_dist, cluster_labels, num_centroids, final_sample_num, iters=1, w=1, momentum=0.5, horizon_dist=None, alpha=1, verbose=False):
    selection_regularizer = torch.zeros_like(neighbors_dist)
    selected_ids_comparison_mask = F.one_hot(
        cluster_labels, num_classes=final_sample_num)
    for _ in tqdm(range(iters)):
        selected_inds = []
        for cls_ind in range(num_centroids):
            if len(selected_inds) == final_sample_num:
                break
            match_arr = cluster_labels == cls_ind
            match = torch.where(match_arr)[0]
            if len(match) == 0:
                continue

            # Scores in the selection process
            scores = 1 / neighbors_dist[match_arr] - \
                w * selection_regularizer[match_arr]
            min_dist_ind = scores.argmax()
            selected_inds.append(match[min_dist_ind])

        selected_inds = np.array(selected_inds)
        selected_data = data[selected_inds]
        # This is square distances: (N_full, N_selected)
        # data: (N_full, 1, dim)
        # selected_data: (1, N_selected, dim)
        new_selection_regularizer = (
            (data[:, None, :] - selected_data[None, :, :]) ** 2).sum(dim=-1)

        if verbose:
            logger.info(
                f"Max: {new_selection_regularizer.max()} Mean: {new_selection_regularizer.mean()}")

        # Distance to the instance within the same cluster should be ignored
        new_selection_regularizer = (1 - selected_ids_comparison_mask) * \
            new_selection_regularizer + selected_ids_comparison_mask * 1e10

        assert not torch.any(new_selection_regularizer == 0), "{}".format(
            torch.where(new_selection_regularizer == 0))

        if verbose:
            logger.info(f"Min: {new_selection_regularizer.min()}")

        # If it is outside of horizon dist (square distance), than we ignore it.
        if horizon_dist is not None:
            new_selection_regularizer[new_selection_regularizer >=
                                      horizon_dist] = 1e10

        # selection_regularizer: N_full
        new_selection_regularizer = (
            1 / new_selection_regularizer ** alpha).sum(dim=1)

        selection_regularizer = selection_regularizer * \
            momentum + new_selection_regularizer * (1 - momentum)

    assert len(
        selected_inds) == final_sample_num, f"{len(selected_inds)} != {final_sample_num}"
    return selected_inds


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class PRETRAIN_CIFAR10(datasets.CIFAR10):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {'image': image, 'target': index of target class, 'meta': dict}
        """
        img, target = self.data[index], self.targets[index]
        img_size = (img.shape[0], img.shape[1])
        img = Image.fromarray(img)
        class_name = self.classes[target]

        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': target, 'meta': {
            'im_size': img_size, 'index': index, 'class_name': class_name}}

        return out


class PRETRAIN_CIFAR100(datasets.CIFAR100):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {'image': image, 'target': index of target class, 'meta': dict}
        """
        img, target = self.data[index], self.targets[index]
        img_size = (img.shape[0], img.shape[1])
        img = Image.fromarray(img)
        class_name = self.classes[target]

        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': target, 'meta': {
            'im_size': img_size, 'index': index, 'class_name': class_name}}

        return out


def train_dataset_cifar(transform_name):
    if transform_name == "FixMatch-cifar10" or transform_name == "SCAN-cifar10" or transform_name == "FixMatch-cifar100" or transform_name == "SCAN-cifar100":
        normalization_kwargs = normalization_kwargs_dict[transform_name]
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            Augment(4),
            transforms.ToTensor(),
            transforms.Normalize(**normalization_kwargs),
            Cutout(
                n_holes=1,
                length=16,
                random=True)])
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**normalization_kwargs)])

        train_transforms = {
            'standard': transform_val, 'augment': transform_train}
    elif transform_name == "CLD-cifar10" or transform_name == "CLD-cifar100":
        # CLD uses MoCov2's aug: similar to SimCLR
        normalization_kwargs = normalization_kwargs_dict[transform_name]
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**normalization_kwargs),
        ])

        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**normalization_kwargs)])
    else:
        raise ValueError(f"Unsupported transform type: {transform_name}")

    if cfg.DATASET.NAME == "cifar10":
        # Transform is set on the wrapper if load_local_global_dataset is True
        train_dataset_cifar = datasets.CIFAR10(
            root=cfg.DATASET.ROOT_DIR, train=True, transform=transform_train, download=True)

        val_dataset = datasets.CIFAR10(
            root=cfg.DATASET.ROOT_DIR, train=False, transform=transform_val, download=True)
    elif cfg.DATASET.NAME == "cifar100":
        train_dataset_cifar = datasets.CIFAR100(
            root=cfg.DATASET.ROOT_DIR, train=True, transform=transform_train, download=True)

        val_dataset = datasets.CIFAR100(
            root=cfg.DATASET.ROOT_DIR, train=False, transform=transform_val, download=True)

    return train_dataset_cifar, val_dataset


# Memory bank on CIFAR
def train_memory_cifar(root_dir, cifar100, transform_name, batch_size=128, workers=2, with_val=False):
    # Note that CLD uses the same normalization for CIFAR 10 and CIFAR 100

    transform_test = get_transform(transform_name)

    if cifar100:
        train_memory_dataset = datasets.CIFAR100(root=root_dir, train=True,
                                                 download=True, transform=transform_test)
        if with_val:
            val_memory_dataset = datasets.CIFAR100(root=root_dir, train=False,
                                                   download=True, transform=transform_test)
    else:
        train_memory_dataset = datasets.CIFAR10(root=root_dir, train=True,
                                                download=True, transform=transform_test)
        if with_val:
            val_memory_dataset = datasets.CIFAR10(root=root_dir, train=False,
                                                  download=True, transform=transform_test)

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
