import os
import time

import numpy as np
import torch
import torch.nn as nn

import torchvision.transforms as transforms

from pykeops.torch import LazyTensor
from tqdm import tqdm
from copy import deepcopy
import inspect


from .config_utils import cfg, logger

# Credit: https://github.com/kekmodel/FixMatch-pytorch


class ModelEMA(object):
    def __init__(self, model, decay):
        self.ema = deepcopy(model)
        self.ema.cuda()
        self.ema.eval()
        self.decay = decay
        self.ema_has_module = hasattr(self.ema, 'module')
        # Fix EMA. https://github.com/valencebond/FixMatch_pytorch thank you!
        self.param_keys = [k for k, _ in self.ema.named_parameters()]
        self.buffer_keys = [k for k, _ in self.ema.named_buffers()]
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()
            for k in self.param_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                model_v = msd[j].detach()
                ema_v = esd[k]
                esd[k].copy_(ema_v * self.decay + (1. - self.decay) * model_v)

            for k in self.buffer_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                esd[k].copy_(msd[j])


def seed_everything(seed):
    if seed is None:
        return

    import os
    import random

    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def save_npy(filename, content):
    if cfg.SKIP_SAVE:
        return

    path = os.path.join(cfg.RUN_DIR, filename)
    if not os.path.exists(path):
        print("Numpy file saved to: {}".format(path))
        np.save(path, content)
    else:
        logger.warning(
            "File exists: {}. Not overwriting (if the file is stale, please save manually).".format(path))


def load_npy(filename, allow_pickle=False):
    path = os.path.join(cfg.RUN_DIR, filename)
    logger.info("Loading saved file from " + path)
    return np.load(path, allow_pickle=allow_pickle)


normalization_kwargs_dict = {
    "CLD-cifar10": dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    "CLD-cifar100": dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    "FixMatch-cifar10": dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
    "FixMatch-cifar100": dict(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
    "SCAN-cifar10": dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    "SCAN-cifar100": dict(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),

    "imagenet": dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    "imagenet100": dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
}


def get_transform(transform_name):
    if transform_name in normalization_kwargs_dict.keys():
        normalization_kwargs = normalization_kwargs_dict[transform_name]
    else:
        raise ValueError(f"Unsupported transform type: {transform_name}")

    if "imagenet" in transform_name:
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(**normalization_kwargs),
        ])
    else:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**normalization_kwargs),
        ])

    return transform_test


def single_model(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def get_feats_list(model, train_memory_loader, CLIP=False, feat_dim=None, recompute=False, dataparallel=False, force_no_extra_kwargs=False, **kwargs):
    if recompute:
        if CLIP:
            model = model.visual
        if dataparallel:
            model_parallel = nn.DataParallel(model)
        else:
            model_parallel = model

        if feat_dim is None:
            feat_dim = 512 if CLIP else 768
            # feat_dim = 512 if CLIP else 128

        feats_list = np.zeros(
            (len(train_memory_loader.dataset), feat_dim), dtype=np.float)
        targets_list = np.zeros(
            len(train_memory_loader.dataset), dtype=np.long)
        with torch.no_grad():
            ptr = 0
            for images, targets in tqdm(train_memory_loader):
                images = images.cuda(non_blocking=True)
                if CLIP:
                    images = images.half()
                    feat = model_parallel(images, **kwargs).cpu().numpy()
                else:
                    if force_no_extra_kwargs:
                        feat = model_parallel(images, **kwargs).cpu().numpy()
                    else:
                        feat = model_parallel(
                            images, get_low_dim_feat=True, **kwargs).cpu().numpy()
                feats_list[ptr:ptr + images.size(0)] = feat
                targets_list[ptr:ptr + images.size(0)] = targets.numpy()
                ptr += images.size(0)

        assert np.all(targets_list == np.array(
            train_memory_loader.dataset.targets))
        save_npy("memory_feats_list.npy", feats_list)
    else:
        feats_list = load_npy("memory_feats_list.npy")
    feats_list = torch.tensor(feats_list).float()
    print("feats_list:", feats_list.shape)

    return feats_list

# Credit: https://www.kernel-operations.io/keops/_auto_tutorials/knn/plot_knn_torch.html


def kNN(x_train, x_test, K=20):
    assert len(x_train.shape) == 2
    assert len(x_test.shape) == 2

    start = time.time()  # Benchmark:

    X_i = LazyTensor(x_test[:, None, :])  # (N1, 1, M) test set
    X_j = LazyTensor(x_train[None, :, :])  # (1, N2, M) train set
    D_ij = ((X_i - X_j) ** 2).sum(
        -1
    )  # (N1, N2) symbolic matrix of squared L2 distances

    # Samples <-> Dataset, (N_test, K)
    d_knn, ind_knn = D_ij.Kmin_argKmin(K, dim=1, backend="GPU")

    # torch.cuda.synchronize()
    end = time.time()

    total_time = end - start
    print("Running time: {:.2f}s".format(total_time))

    return ind_knn, d_knn


def partitioned_kNN(feats_list, K=20, recompute=False, partitions_size=130000, verify=False):
    suffix = "" if K == 20 else "_{}".format(K)
    if recompute:
        partitions = int(np.ceil(feats_list.shape[0] / partitions_size))

        print("Partitions:", partitions)

        # Assume the last partition has at least K elements

        ind_knns = torch.zeros(
            (feats_list.size(0), partitions * K), dtype=torch.long)
        d_knns = torch.zeros(
            (feats_list.size(0), partitions * K), dtype=torch.float)

        def get_sampled_data(ind):
            return feats_list[ind * partitions_size: (ind + 1) * partitions_size]

        for ind_i in range(partitions):  # ind_i: train dimension
            for ind_j in range(partitions):  # ind_j: test dimension
                print("Running with indices: {}, {}".format(ind_i, ind_j))
                x_train = get_sampled_data(ind_i).cuda()
                x_test = get_sampled_data(ind_j).cuda()

                ind_knn, d_knn = kNN(x_train, x_test, K=K)
                # ind_knn, d_knn: test dimension, K (indices: train dimension)
                ind_knns[ind_j * partitions_size: (ind_j + 1) * partitions_size, ind_i * K: (ind_i + 1) * K] = \
                    ind_i * partitions_size + ind_knn.cpu()
                d_knns[ind_j * partitions_size: (ind_j + 1) * partitions_size,
                       ind_i * K: (ind_i + 1) * K] = d_knn.cpu()

                del ind_knn, d_knn, x_train, x_test

        d_sorted_inds = d_knns.argsort(dim=1)
        d_selected_inds = d_sorted_inds[:, :K]
        ind_knns_selected = torch.gather(
            ind_knns, dim=1, index=d_selected_inds)
        d_knns_selected = torch.gather(d_knns, dim=1, index=d_selected_inds)
        d_knns = d_knns_selected
        ind_knns = ind_knns_selected

        del ind_knns_selected, d_knns_selected

        if verify:  # Verification
            ind_knns_target, d_knns_target = kNN(
                feats_list.cuda(), feats_list.cuda())
            ind_knns_target = ind_knns_target.cpu()
            d_knns_target = d_knns_target.cpu()
            assert torch.all(d_knns == d_knns_target)
            # The ids may differ, but as long as the distance of the selected indices is correct:
            # assert torch.all(ind_knns != ind_knns_target)
            if not torch.all(ind_knns == ind_knns_target):
                def dist(a, b):
                    return torch.sum((a - b)**2)

                for dim1, dim2 in zip(*torch.where(ind_knns != ind_knns_target)):
                    dist1 = dist(feats_list[dim1],
                                 feats_list[ind_knns[dim1][dim2]])
                    dist2 = dist(
                        feats_list[dim1], feats_list[ind_knns_target[dim1][dim2]])
                    assert torch.isclose(
                        dist1, dist2), "{} != {}".format(dist1, dist2)
        save_npy("d_knns{}.npy".format(suffix), d_knns.cpu().numpy())
        save_npy("ind_knns{}.npy".format(suffix), ind_knns.cpu().numpy())
    else:
        d_knns = torch.tensor(load_npy("d_knns{}.npy".format(suffix)))
        ind_knns = torch.tensor(load_npy("ind_knns{}.npy".format(suffix)))

    return d_knns, ind_knns

# Credit: https://www.kernel-operations.io/keops/_auto_tutorials/kmeans/plot_kmeans_torch.html


def KMeans(x, seed, K=10, Niter=10, init_inds=None, verbose=True, force_no_lazy_tensor=False):
    """Implements Lloyd's algorithm for the Euclidean metric."""

    start = time.time()
    N, D = x.shape  # Number of samples, dimension of the ambient space

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    if init_inds is None:
        print("Use no init indices")
        r = torch.randperm(x.shape[0])[:K]
    else:
        print("Using init indices")
        assert K <= init_inds.shape[0]
        r = torch.randperm(init_inds.shape[0])[:K]
        r = init_inds[r]

    if verbose:
        print("Init indices {}".format(r.numpy()))

    if force_no_lazy_tensor:  # For deterministic behavior
        print("No lazy tensor")
        def _LazyTensor(x): return x
        kwargs = {
        }
    else:
        _LazyTensor = LazyTensor
        kwargs = {
            "backend": "GPU"
        }

    assert r.shape[0] == K, "{} != {}".format(r.shape[0], K)

    c = x[r, :].clone()  # Simplistic initialization for the centroids

    x_i = _LazyTensor(x.view(N, 1, D))

    c_j = _LazyTensor(c.view(1, K, D))  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for _ in tqdm(range(Niter)):
        # E step: assign points to the closest cluster -------------------------
        # Perform this op in halves to avoid using up GPU memory:

        # (N, K) symbolic squared distances
        D_ij = ((x_i - c_j) ** 2).sum(-1, **kwargs)
        # Points -> Nearest cluster
        cl = D_ij.argmin(dim=1, **kwargs).long().view(-1)

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        c /= Ncl  # in-place division to compute the average

    if verbose:  # Fancy display -----------------------------------------------
        # if use_cuda:
        #     torch.cuda.synchronize()
        end = time.time()
        print(
            f"K-means for the Euclidean metric with {N:,} points in dimension {D:,}, K = {K:,}:"
        )
        print(
            "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                Niter, end - start, Niter, (end - start) / Niter
            )
        )

    return cl, c


def run_kMeans(feats_list, num_centroids, final_sample_num, Niter, recompute=False, use_cuda=True, seed=None, force_no_lazy_tensor=False, save=True):
    seed_suffix = "_{}".format(seed) if seed is not None else ""
    if recompute:
        # use_cuda: pre-copy data to GPU instead of getting data when running
        cluster_labels, centroids = KMeans(feats_list.cuda() if use_cuda else feats_list, seed=seed,
                                           K=num_centroids, Niter=Niter, verbose=True, force_no_lazy_tensor=force_no_lazy_tensor)
        cluster_labels, centroids = cluster_labels.cpu(), centroids.cpu()

        inds, cnts = torch.unique(cluster_labels, return_counts=True)
        logger.info(
            f"Num of clusters: {len(inds)} counts: {cnts} min: {cnts.min().item()} max: {cnts.max().item()}")
        if save:
            save_npy("cluster_labels_{}{}.npy".format(
                final_sample_num, seed_suffix), cluster_labels.numpy())
            save_npy("centroids_{}{}.npy".format(
                final_sample_num, seed_suffix), centroids.numpy())
    else:
        cluster_labels = torch.tensor(
            load_npy("cluster_labels_{}{}.npy".format(final_sample_num, seed_suffix)))
        centroids = torch.tensor(
            load_npy("centroids_{}{}.npy".format(final_sample_num, seed_suffix)))

    return cluster_labels, centroids


def get_selection_without_reg(cluster_labels, neighbors_dist, centroid_ordering, final_sample_num):
    selected_indices = []
    cluster_labels_cuda = cluster_labels.cuda()
    neighbors_dist_cuda = neighbors_dist.cuda()
    if isinstance(centroid_ordering, int):
        # This is for compatibility: the argument is num_centroids and we use order from 0
        centroid_ordering = range(centroid_ordering)
    for cls_ind in tqdm(centroid_ordering):
        match_arr = cluster_labels_cuda == cls_ind
        if not torch.any(match_arr):
            continue
        # Select the instance with lowest distance:
        min_dist_ind = neighbors_dist_cuda[match_arr].argmin()
        # Select outliers:
        # min_dist_ind = neighbors_dist_cuda[match_arr].argmax()
        selected_indices.append(
            torch.where(match_arr)[0][min_dist_ind].item())
    selected_indices = np.array(
        selected_indices)

    del cluster_labels_cuda
    del neighbors_dist_cuda
    print("selected_indices size (not cut to final_sample_num):",
          selected_indices.shape)
    # if selected_indices.shape[0] < final_sample_num:
    #     warnings.warn("Insufficient data: expected: {}, actual: {}".format(final_sample_num, selected_indices.shape[0]))
    assert selected_indices.shape[0] >= final_sample_num, "Insufficient data: expected: {}, actual: {}".format(
        final_sample_num, selected_indices.shape[0])
    # We only retain the first final_sample_num, so the centroid ordering matters if the truncation happens
    selected_indices = selected_indices[:final_sample_num]

    return selected_indices


def get_selection(selection_fn, *args, seed=None, recompute=True, save=True, pass_seed=False, **kwargs):
    # This is deterministic, and the seed is just for naming the file to save and load
    seed_suffix = "_{}".format(seed) if seed is not None else ""
    final_sample_num = kwargs["final_sample_num"]
    save_filename = "selected_indices_{}{}.npy".format(
        final_sample_num, seed_suffix)



    if recompute:
        if pass_seed:
            selected_inds = selection_fn(*args, seed=seed, **kwargs)
        else:
            argspec = inspect.getfullargspec(selection_fn)
            selected_inds = selection_fn(*args, **kwargs)

        if save:
            save_npy(save_filename, selected_inds)
    else:
        selected_inds = load_npy(save_filename)

    return selected_inds
