from .config_utils import print_r
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import collections
# from torch._six import string_classes
string_classes = (str,)

# Credit to PAWS: https://github.com/facebookresearch/suncet/blob/main/src/losses.py
def sharpen(p, T):  # T: sharpen temperature
    sharp_p = p ** (1. / T)
    sharp_p = sharp_p / torch.sum(sharp_p, dim=1, keepdim=True)
    return sharp_p


class OursLossLocal(nn.Module):
    def __init__(self, num_classes, num_heads, momentum=None, adjustment_weight=None, sharpen_temperature=None):
        super(OursLossLocal, self).__init__()
        self.momentum = momentum

        self.adjustment_weight = adjustment_weight

        self.num_heads = num_heads

        self.register_buffer("prob_ema", torch.ones(
            (num_heads, num_classes)) / num_classes)

        self.sharpen_temperature = sharpen_temperature

    def forward(self, head_id, anchors, neighbors):
        # This is ours v2 with multi_headed prob_ema support
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]

        output:
            - Loss
        """
        # Softmax
        b, n = anchors.size()
        head_prob_ema = self.prob_ema[head_id]
        neighbors_adjusted = neighbors - self.adjustment_weight * \
            torch.log(head_prob_ema).view((1, -1))

        anchors_prob = F.softmax(anchors, dim=1)
        positives_prob = F.softmax(neighbors_adjusted, dim=1)
        log_anchors_prob = F.log_softmax(anchors, dim=1)

        positives_original_prob = F.softmax(neighbors, dim=1)
        head_prob_ema = head_prob_ema * self.momentum + \
            positives_original_prob.detach().mean(dim=0) * (1 - self.momentum)
        head_prob_ema = head_prob_ema / head_prob_ema.sum()

        self.prob_ema[head_id] = head_prob_ema

        consistency_loss = F.kl_div(log_anchors_prob, sharpen(
            positives_prob.detach(), T=self.sharpen_temperature), reduction="batchmean")

        # Total loss
        total_loss = consistency_loss

        return total_loss


class OursLossGlobal(nn.Module):
    # From ConfidenceBasedCE
    def __init__(self, threshold, reweight, num_classes, num_heads, mean_outside_mask=False, use_count_ema=False, momentum=0., data_len=None, reweight_renorm=False):
        super(OursLossGlobal, self).__init__()
        self.threshold = threshold
        self.reweight = reweight
        # setting reweight_renorm to True ignores reweight
        self.reweight_renorm = reweight_renorm

        if self.reweight_renorm:
            print("Reweight renorm is enabled")
        else:
            print("Reweight renorm is not enabled")

        self.mean_outside_mask = mean_outside_mask
        self.use_count_ema = use_count_ema

        self.num_classes = num_classes
        self.num_heads = num_heads

        self.momentum = momentum

        if use_count_ema:
            print("Data length:", data_len)
            self.data_len = data_len
            self.register_buffer("count_ema", torch.ones(
                (num_heads, num_classes)) / num_classes)
        self.register_buffer("num_counts", torch.zeros(1, dtype=torch.long))

    # Equivalent to: https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
    # With one-hot target
    def kl_div_loss(self, input, target, mask, weight, mean_outside_mask):
        if torch.all(mask == 0):
            # Return 0 as loss if nothing is in mask
            return torch.tensor(0., device=input.device)

        b = input.shape[0]

        # Select samples that pass the confidence threshold
        input = torch.masked_select(
            input, mask.view(b, 1)).view((-1, input.shape[1]))
        target = torch.masked_select(target, mask)

        log_prob = -F.log_softmax(input, dim=1)
        if weight is not None:
            # Weighted KL divergence
            log_prob = log_prob * weight.view((1, -1))
        loss = torch.gather(log_prob, 1, target.view((-1, 1))).view(-1)

        if mean_outside_mask:
            # Normalize by a constant (batch size)
            return loss.sum(dim=0) / b
        else:
            if weight is not None:
                # Take care of weighted sum
                weight_sum = weight[target].sum(dim=0)
                return (loss / weight_sum).sum(dim=0)
            else:
                return loss.mean(dim=0)

    def forward(self, head_id, anchors_weak, anchors_strong):
        """
        Loss function during self-labeling

        input: logits for original samples and for its strong augmentations 
        output: cross entropy 
        """
        # Retrieve target and mask based on weakly augmentated anchors

        weak_anchors_prob = F.softmax(anchors_weak, dim=1)

        max_prob, target = torch.max(weak_anchors_prob, dim=1)
        mask = max_prob > self.threshold
        b, c = weak_anchors_prob.size()
        target_masked = torch.masked_select(target, mask.squeeze())
        n = target_masked.size(0)

        if self.use_count_ema:
            with torch.no_grad():
                head_count_ema = self.count_ema[head_id]

                # Normalized and adjusted with data_len
                count_in_batch = torch.bincount(
                    target_masked, minlength=c) / n * self.data_len
                head_count_ema = head_count_ema * self.momentum + \
                    count_in_batch * (1 - self.momentum)
                self.count_ema[head_id] = head_count_ema

        if head_id == 0:
            self.num_counts += 1

        # Class balancing weights
        # This is also used for debug purpose

        # reweight_renorm is equivalent to reweight when mean_outside_mask is False
        if self.reweight_renorm:
            idx, counts = torch.unique(target_masked, return_counts=True)
            # if self.use_count_ema:
            #     print("WARNING: count EMA used with class balancing")
            freq = float(n) / len(idx) / counts.float()
            weight = torch.ones(c).cuda()
            weight[idx] = freq
        elif self.reweight:
            idx, counts = torch.unique(target_masked, return_counts=True)
            if self.use_count_ema:
                print("WARNING: count EMA used with class balancing")
            freq = 1/(counts.float()/n)
            weight = torch.ones(c).cuda()
            weight[idx] = freq
        else:
            weight = None

        # Loss

        loss = self.kl_div_loss(input=anchors_strong, target=target, mask=mask,
                                weight=weight, mean_outside_mask=self.mean_outside_mask)

        if head_id == 0 and self.num_counts % 200 == 1:
            with torch.no_grad():
                idx, counts = torch.unique(target_masked, return_counts=True)
            if self.use_count_ema:
                print_r("use_count_ema max: {:.3f}, min: {:.3f}, median: {:.3f}, mean: {:.3f}".format(head_count_ema.max().item(),
                                                                                                      head_count_ema.min().item(), torch.median(head_count_ema).item(), head_count_ema.mean().item()))
            print_r("weak_anchors_prob, mean across batch (from weak anchor of global loss): {}".format(
                weak_anchors_prob.detach().mean(dim=0)))
            print_r("Mask: {} / {} ({:.2f}%)".format(mask.sum(),
                                                     mask.shape[0], mask.sum() * 100. / mask.shape[0]))
            print_r("idx: {}, counts: {}".format(idx, counts))

            if True:  # Verbose: print max confidence of each class
                m = torch.zeros((self.num_classes,))
                for i in range(self.num_classes):
                    v = max_prob[target == i]
                    if len(v):
                        m[i] = v.max()

                print_r("Max of each cluster: {}".format(m))

        return loss


class NormedLinear(nn.Module):
    # (1 / 0.07) is self-supervised learning value. 40 makes it easier to meet the confidence threshold
    def __init__(self, in_features: int, out_features: int, scale_init: int = 40, head_id=None) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        # You need to initialize on your own
        self.weight.data.fill_(0.0)

        # Cannot learn scale to prevent collapse
        self.scale = scale_init  # nn.Parameter(torch.ones((1,)) * scale_init)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Use normalize with eps

        # input: N x D
        # output: Dout (cluster num) x D
        return F.linear(F.normalize(input, dim=1), F.normalize(self.weight, dim=1)) * self.scale

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )


"""
Reference for the code below:
Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""


class ContrastiveModel(nn.Module):
    def __init__(self, backbone, head='mlp', backbone_dim=2048, features_dim=128):
        super(ContrastiveModel, self).__init__()
        self.backbone = backbone
        self.backbone_dim = backbone_dim
        self.head = head

        if head == 'linear':
            self.contrastive_head = nn.Linear(self.backbone_dim, features_dim)

        elif head == 'mlp':
            self.contrastive_head = nn.Sequential(
                nn.Linear(self.backbone_dim, self.backbone_dim),
                nn.ReLU(), nn.Linear(self.backbone_dim, features_dim))

        else:
            raise ValueError('Invalid head {}'.format(head))

    def forward(self, x):
        features = self.contrastive_head(self.backbone(x))
        features = F.normalize(features, dim=1)
        return features


class ClusteringModel(nn.Module):
    def __init__(self, backbone, nclusters, backbone_dim=512, nheads=1, normed=False, **kwargs):
        super(ClusteringModel, self).__init__()
        self.backbone = backbone
        self.backbone_dim = backbone_dim
        self.nheads = nheads
        assert(isinstance(self.nheads, int))
        assert(self.nheads > 0)
        if normed:
            import warnings
            warnings.warn("Using normed clustering model")

        if normed:
            self.cluster_head = nn.ModuleList(
                [NormedLinear(self.backbone_dim, nclusters, head_id=i, **kwargs) for i in range(self.nheads)])
        else:
            self.cluster_head = nn.ModuleList(
                [nn.Linear(self.backbone_dim, nclusters, **kwargs) for _ in range(self.nheads)])

    def forward(self, x):
        x = self.backbone(x)
        out = [cluster_head(x) for cluster_head in self.cluster_head]

        return out


def collate_custom(batch):
    """ Custom collate function """

    if isinstance(batch[0], np.int64):
        return np.stack(batch, 0)

    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)

    elif isinstance(batch[0], np.ndarray):
        return np.stack(batch, 0)

    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)

    elif isinstance(batch[0], float):
        return torch.FloatTensor(batch)

    elif isinstance(batch[0], string_classes):
        return batch

    elif isinstance(batch[0], collections.Mapping):
        batch_modified = {key: collate_custom(
            [d[key] for d in batch]) for key in batch[0] if key.find('idx') < 0}
        return batch_modified

    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [collate_custom(samples) for samples in transposed]

    raise TypeError(('Type is {}'.format(type(batch[0]))))


class NeighborsDataset(Dataset):
    """ 
    NeighborsDataset
    Returns an image with one of its neighbors.
    """

    def __init__(self, dataset, indices, num_neighbors=None):
        super(NeighborsDataset, self).__init__()
        transform = dataset.transform

        if isinstance(transform, dict):
            self.neighbor_transform = transform['augment']
        else:
            self.neighbor_transform = transform

        dataset.transform = None
        self.dataset = dataset
        # Nearest neighbor indices (np.array  [len(dataset) x k])
        self.indices = indices
        if num_neighbors is not None:
            self.indices = self.indices[:, :num_neighbors+1]
        assert(self.indices.shape[0] == len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = {}
        neighbor_index = np.random.choice(self.indices[index], 1)[0]
        neighbor = self.dataset.__getitem__(neighbor_index)
        neighbor['image'] = self.neighbor_transform(neighbor['image'])
        output['neighbor'] = neighbor['image']
        # output['possible_neighbors'] = torch.from_numpy(self.indices[index])
        return output


class LocalGlobalDataset(Dataset):
    def __init__(self, dataset, transform, neighbors_indices, num_neighbors, add_neighbors=True):
        super().__init__()
        assert dataset.transform is None, f"Expect the dataset to be wrapped to have None transform but encounter {dataset.transform}"

        # Note that we have three transforms set in this dataset: neighbor_transform, image_transform, augmentation_transform

        self.dataset = dataset

        # This could be modified from outside of the dataset, so we always initialize neighbor dataset
        self.add_neighbors = add_neighbors

        # Neighbors use strong transform
        self.neighbor_init(
            transform=transform["augment"], neighbors_indices=neighbors_indices, num_neighbors=num_neighbors)

        # Use both weak and strong augmentation
        self.aug_init(transform=transform)

    def __len__(self):
        return len(self.dataset)

    def neighbor_init(self, transform, neighbors_indices, num_neighbors):
        if isinstance(transform, dict):
            self.neighbor_transform = transform['augment']
        else:
            self.neighbor_transform = transform

        # Nearest neighbor indices (np.array  [len(dataset) x k])
        self.neighbors_indices = neighbors_indices
        if num_neighbors is not None:
            assert self.neighbors_indices.shape[1] >= num_neighbors + \
                1, f"The NN indices with dimension {self.neighbors_indices.shape[1]} does not allow num_neighbors to be set to {num_neighbors}"
            self.neighbors_indices = self.neighbors_indices[:,
                                                            :num_neighbors+1]

        assert self.neighbors_indices.shape[0] == len(
            self.dataset), f"{self.neighbors_indices.shape[0]} != {len(self.dataset)}"

    def aug_init(self, transform):
        if isinstance(transform, dict):
            self.image_transform = transform['standard']
            self.augmentation_transform = transform['augment']
        else:
            self.image_transform = transform
            self.augmentation_transform = transform

    def get_augmented_dataset_item(self, index):
        sample = self.dataset.__getitem__(index)
        image = sample['image']

        sample['image'] = self.image_transform(image)
        sample['image_augmented'] = self.augmentation_transform(image)

        return sample

    def get_neighbor_dataset_item(self, index):
        output = {}
        neighbor_index = np.random.choice(self.neighbors_indices[index], 1)[0]
        neighbor = self.dataset.__getitem__(neighbor_index)
        neighbor['image'] = self.neighbor_transform(neighbor['image'])
        output['neighbor'] = neighbor['image']
        # output['possible_neighbors'] = torch.from_numpy(self.indices[index])
        return output

    def __getitem__(self, index):
        aug_output = self.get_augmented_dataset_item(index)

        if self.add_neighbors:
            neighbors_output = self.get_neighbor_dataset_item(index)
            return {
                "neighbors": neighbors_output,
                "aug": aug_output
            }
        else:
            return {
                "aug": aug_output
            }


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
