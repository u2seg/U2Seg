# %%
import os
os.environ["USL_MODE"] = "USLT_PRETRAIN"

from tqdm import tqdm
from utils import cfg, logger, print_b, print_y, seed_everything
import utils
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from utils import uslt_utils
from utils.dataloader import MultiEpochsDataLoader
from sklearn import metrics
from utils.uslt_utils import AverageMeter, ProgressMeter

# %%
utils.init(default_config_file="configs/imagenet_usl-t_pretrain.yaml")
logger.info(cfg)
seed_everything(cfg.SEED)
if cfg.BACKUP_PYTHON_FILE and not cfg.EVAL_ONLY:
    import shutil
    shutil.copy(__file__, cfg.RUN_DIR)

# %%
print_b("Loading dataset")
assert cfg.DATASET.NAME in [
    "imagenet100", "imagenet"], f"{cfg.DATASET.NAME} is not imagenet100 or imagenet"
imagenet100 = cfg.DATASET.NAME == "imagenet100"
num_classes = 100 if imagenet100 else 1000

train_dataset, val_dataset, memory_bank_dataset = utils.train_dataset_imagenet(
    transform_name=cfg.DATASET.TRANSFORM_NAME, load_local_global_dataset=True, add_memory_bank_dataset=True)

train_dataloader = MultiEpochsDataLoader(train_dataset, num_workers=cfg.DATALOADER.WORKERS,
                              batch_size=cfg.DATALOADER.BATCH_SIZE, pin_memory=True, collate_fn=uslt_utils.collate_custom,
                              drop_last=True, shuffle=True)
val_dataloader = MultiEpochsDataLoader(val_dataset, num_workers=cfg.DATALOADER.WORKERS,
                            batch_size=cfg.DATALOADER.BATCH_SIZE, pin_memory=True, 
                            drop_last=False, shuffle=False)
memory_bank_dataloader = MultiEpochsDataLoader(memory_bank_dataset, num_workers=cfg.DATALOADER.WORKERS,
                            batch_size=cfg.DATALOADER.BATCH_SIZE, pin_memory=True, 
                            drop_last=False, shuffle=False)

val_targets = torch.tensor(val_dataset.targets)
val_targets.shape

# %%
print_b("Loading model")

checkpoint = torch.load(cfg.MODEL.PRETRAIN_PATH, map_location="cpu")

num_heads = 3
num_labeled = cfg.USLT_PRETRAIN.NUM_SELECTED_SAMPLES

backbone = models.__dict__[cfg.MODEL.ARCH](pretrained=False)
backbone.fc = nn.Identity()
# BACKBONE_DIM is used with ContrastiveModel
backbone = utils.ContrastiveModel(
    backbone, head='mlp', features_dim=128, backbone_dim=cfg.MODEL.BACKBONE_DIM)
# Apply L2 Norm with 128 dim feat (not BACKBONE_DIM as in smaller datasets)
model = utils.ClusteringModel(backbone, nclusters=num_labeled, normed=True,
                              nheads=num_heads, backbone_dim=128).cuda()

logger.info("Model: " + str(model))
# Only load backbone weights
mismatch = backbone.load_state_dict(
    utils.single_model(checkpoint), strict=False)
logger.warning(
    f"Key mismatches: {mismatch}")

# %%
if cfg.USLT_PRETRAIN.EMA_DECAY > 0:
    print_b("Loading EMA")
    from utils import ModelEMA
    ema_model = ModelEMA(model, cfg.USLT_PRETRAIN.EMA_DECAY)
else:
    ema_model = None

# %%
print_b("Initializing optimizer")
torch.backends.cudnn.benchmark = True
if cfg.USLT_PRETRAIN.UPDATE_HEAD_ONLY:
    for (name, param) in model.named_parameters():
        if "cluster_head" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
else:
    for (name, param) in model.named_parameters():
        param.requires_grad = True

params = [param for _, param in model.named_parameters()
          if param.requires_grad]
param_names = [name for name, param in model.named_parameters()
               if param.requires_grad]

logger.info(f"Allow grads on: {param_names}")

optimizer = torch.optim.__dict__[cfg.OPTIMIZER.NAME](params,
                                                     lr=cfg.OPTIMIZER.LR,
                                                     weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY)

# %%
print_b("Initializing losses")

criterion_local = utils.OursLossLocal(
    num_classes=num_labeled, num_heads=num_heads, adjustment_weight=cfg.USLT_PRETRAIN.ADJUSTMENT_WEIGHT, momentum=0.5, sharpen_temperature=cfg.USLT_PRETRAIN.SHARPEN_TEMPERATURE).cuda()
criterion_global = utils.OursLossGlobal(num_classes=num_labeled, num_heads=num_heads, use_count_ema=True, data_len=len(train_dataset),
                                        threshold=cfg.USLT_PRETRAIN.CONFIDENCE_THRESHOLD, reweight=cfg.USLT_PRETRAIN.REWEIGHT, reweight_renorm=cfg.USLT_PRETRAIN.REWEIGHT_RENORM, momentum=0.9, mean_outside_mask=cfg.USLT_PRETRAIN.MEAN_OUTSIDE_MASK).cuda()
criterion = {"global": criterion_global, "local": criterion_local}
print_b(criterion)

# %%

@torch.no_grad()
def get_predictions(dataloader, model, num_heads=3):
    # Make predictions on a dataset with neighbors
    model.eval()

    predictions = [[] for _ in range(num_heads)]
    probs = [[] for _ in range(num_heads)]
    targets = []

    for i, (images, target) in enumerate(tqdm(dataloader)):
        images = images.cuda(non_blocking=True)
        output = model(images)

        for i, output_i in enumerate(output):
            predictions[i].append(torch.argmax(output_i, dim=1).cpu())
            probs[i].append(F.softmax(output_i, dim=1).cpu())
        targets.append(target)

    predictions = [torch.cat(pred_, dim=0) for pred_ in predictions]
    probs = [torch.cat(prob_, dim=0) for prob_ in probs]
    targets = torch.cat(targets, dim=0)

    out = [{'predictions': pred_, 'probabilities': prob_, 'targets': targets}
           for pred_, prob_ in zip(predictions, probs)]

    return out

# Credit: https://github.com/facebookresearch/DeeperCluster/blob/d38ada109f8334f6ae4c84a218d79848a936ed6f/src/distributed_kmeans.py#L259-L279


def reassign_cluster(cluster_head, min_ind, max_ind, count_ema_head, eps=1e-7):
    # weight (normed): out_feat, in_feat
    # bias is not included for normed

    max_weight = cluster_head.weight[max_ind]

    pert = torch.randn_like(max_weight) * eps
    cluster_head.weight[min_ind] = max_weight + pert
    cluster_head.weight[max_ind] = max_weight - pert

    original_count = count_ema_head[max_ind]

    # Approximate new count
    count_ema_head[min_ind] = original_count / 2
    count_ema_head[max_ind] = original_count / 2


# %%
print_b("Start training")

checkpoint_save_path = os.path.join(cfg.RUN_DIR, "checkpoint.pth")


def evaluate_predictions(predictions, targets):
    n_correct = 0

    for cluster_ind in torch.unique(predictions):
        cluster_i_mask = predictions == cluster_ind
        # `num_classes` is the number of ground truth classes in targets
        cluster_i_target = torch.bincount(targets[cluster_i_mask])

        # Use max as what this cluser represents
        n_correct += cluster_i_target.max()

    macc = n_correct.item() / float(predictions.shape[0])

    targets_np = targets.numpy()
    predictions_np = predictions.numpy()
    nmi = metrics.normalized_mutual_info_score(
        targets_np, predictions_np)
    ari = metrics.adjusted_rand_score(
        targets_np, predictions_np)

    unique_clusters, cluster_sample_counts = torch.unique(
        predictions, return_counts=True)
    num_non_empty_clusters = len(unique_clusters)

    return macc, nmi, ari, num_non_empty_clusters, cluster_sample_counts.max().item(), cluster_sample_counts.min().item()


def ours_train(train_loader, model, ema_model, criterion, optimizer, epoch, update_head_only=False, cfg=None):
    if epoch == 0:
        print_b('Initializing clustering head')
        # Initial weight assignment by sampling iid random samples from training set to begin with

        model.eval()
        with torch.no_grad():
            sample_counts = 0
            output_dim = model.cluster_head[0].out_features
            samples_needed = model.nheads * output_dim
            original_cluster_head = model.cluster_head
            model.cluster_head = nn.ModuleList([nn.Identity()])
            outputs = []
            for i, batch in enumerate(train_loader):
                batch_aug = batch['aug']
                images = batch_aug['image'].cuda(non_blocking=True)
                output = model(images)[0]

                outputs.append(output)
                sample_counts += output.shape[0]

                if sample_counts >= samples_needed:
                    break
            outputs = torch.cat(outputs, dim=0)[:samples_needed]
            model.cluster_head = original_cluster_head

            for head_id, cluster_head_item in enumerate(model.cluster_head):
                cluster_head_item.weight.data.copy_(
                    outputs[head_id * output_dim: (head_id + 1) * output_dim])

    total_losses = AverageMeter('Total Loss', ':.4')
    global_losses = AverageMeter('Global Loss', ':.4')
    local_losses = AverageMeter('Local Loss', ':.4')

    progress = ProgressMeter(len(train_loader),
                             [total_losses, global_losses, local_losses],
                             prefix="Epoch: [{}]".format(epoch))

    if update_head_only:
        model.eval()  # No need to update BN
    else:
        model.train()  # Update BN

    criterion_local, criterion_global = criterion["local"], criterion["global"]

    add_local = epoch <= cfg.USLT_PRETRAIN.TURN_OFF_LOCAL_LOSS_AFTER_EPOCH
    local_loss_scale = cfg.USLT_PRETRAIN.LOCAL_LOSS_SCALE
    train_loader.dataset.add_neighbors = add_local

    count_ema_min_th, count_ema_max_th = cfg.USLT_PRETRAIN.COUNT_EMA_MIN_TH, cfg.USLT_PRETRAIN.COUNT_EMA_MAX_TH
    reassign_after_steps = cfg.USLT_PRETRAIN.REASSIGN_AFTER_STEPS

    for i, batch in enumerate(train_loader):
        # Forward pass
        batch_aug = batch['aug']

        # Global (k-Means) part
        images = batch_aug['image'].cuda(non_blocking=True)
        images_augmented = batch_aug['image_augmented'].cuda(non_blocking=True)

        with torch.no_grad():
            output = model(images)
        output_augmented = model(images_augmented)

        if epoch >= cfg.USLT_PRETRAIN.GLOBAL_START_EPOCH:  # 0: start from epoch 0
            global_loss_list = []
            for head_id, (images_output_subhead, images_augmented_output_subhead) in enumerate(zip(output, output_augmented)):
                global_loss_item = criterion_global(
                    head_id, images_output_subhead, images_augmented_output_subhead)
                global_loss_list.append(global_loss_item)
        else:
            global_loss_list = [torch.tensor([0.], device=output[0].device)]

        global_losses.update(np.mean([v.item() for v in global_loss_list]))
        global_loss = torch.sum(torch.stack(global_loss_list, dim=0))

        # Local (Neighbors) part
        if add_local:
            batch_neighbors = batch['neighbors']
            neighbors = batch_neighbors['neighbor'].cuda(non_blocking=True)

            anchors_output = output_augmented
            with torch.no_grad():
                neighbors_output = model(neighbors)

            # Loss for every head
            local_loss_list = []

            for head_id, (anchors_output_subhead, neighbors_output_subhead) in enumerate(zip(anchors_output, neighbors_output)):
                local_loss_item = criterion_local(head_id, anchors_output_subhead,
                                                  neighbors_output_subhead)
                local_loss_list.append(local_loss_item)

            # Register the mean loss and backprop the total loss to cover all subheads
            local_losses.update(np.mean([v.item() for v in local_loss_list]))

            total_losses.update(
                np.mean([v.item() for v in global_loss_list]) + np.mean([v.item() for v in local_loss_list]))

            local_loss = local_loss_scale * \
                torch.sum(torch.stack(local_loss_list, dim=0))

        else:
            local_losses.update(0.)

            total_losses.update(np.mean([v.item() for v in global_loss_list]))

            local_loss = 0.

        total_loss = global_loss + local_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if ema_model is not None:  # Apply EMA to update the weights of the network
            ema_model.update(model)

        # Reassignment if distribution counting is enabled in global loss
        if criterion_global.use_count_ema and criterion_global.num_counts > reassign_after_steps:

            with torch.no_grad():
                count_ema = criterion_global.count_ema
                for head_id in range(len(count_ema)):
                    count_ema_head = count_ema[head_id]
                    min_val, min_ind = count_ema_head.min(dim=0)
                    max_val, max_ind = count_ema_head.max(dim=0)

                    if min_val < count_ema_min_th and max_val > count_ema_max_th:
                        cluster_head = model.cluster_head[head_id]
                        if head_id == 0:
                            print_b("Reassigning cluster {} with count {:.3f} to cluster {} with count {:.3f}".format(
                                min_ind, min_val, max_ind, max_val))
                            # import ipdb; ipdb.set_trace()
                        reassign_cluster(cluster_head, min_ind,
                                         max_ind, count_ema_head)

        if i % 10 == 0:
            progress.display(i)
            break


for epoch in range(cfg.EPOCHS):
    print_y('Epoch %d/%d' % (epoch+1, cfg.EPOCHS))

    ours_train(train_dataloader, model, ema_model, criterion, optimizer,
               epoch, update_head_only=cfg.USLT_PRETRAIN.UPDATE_HEAD_ONLY, cfg=cfg)

    print_b('Make predictions on validation set')
    with torch.no_grad():
        for ema_eval in [False, True]:
            if ema_eval:
                if ema_model is None:
                    continue
                val_predictions = get_predictions(
                    val_dataloader, ema_model.ema)
                ema_str = "EMA "
            else:
                val_predictions = get_predictions(val_dataloader, model)
                ema_str = ""
            for ind, val_predictions_head in enumerate(val_predictions):
                val_predictions, val_targets = val_predictions_head[
                    "predictions"], val_predictions_head["targets"]
                macc, nmi, ari, num_non_empty_clusters, count_max, count_min = evaluate_predictions(
                    val_predictions, val_targets)
                print(
                    f"{ema_str}Head {ind}: MACC: {macc * 100.:.2f}, NMI: {nmi * 100.:.2f}, ARI: {ari * 100.:.2f}, Non-Empty (train/val): {num_non_empty_clusters}, Max count: {count_max}, Min count: {count_min}")

    torch.save({
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'ema_model': ema_model.ema.state_dict() if ema_model is not None else None,
        'epoch': epoch + 1,
        "criterion_local": criterion_local.state_dict(),
        "criterion_global": criterion_global.state_dict(),
        "val_predictions": val_predictions,
    }, checkpoint_save_path)

print_b('Make predictions on training set (memory bank)')
with torch.no_grad():
    memory_bank_predictions = get_predictions(memory_bank_dataloader, model)

    for ind, memory_bank_predictions_head in enumerate(memory_bank_predictions):
        memory_bank_predictions, memory_bank_targets = memory_bank_predictions_head[
            "predictions"], memory_bank_predictions_head["targets"]
        macc, nmi, ari, num_non_empty_clusters, count_max, count_min = evaluate_predictions(
            memory_bank_predictions, memory_bank_targets)
        print(
            f"Head {ind}: MACC: {macc * 100.:.2f}, NMI: {nmi * 100.:.2f}, ARI: {ari * 100.:.2f}, Non-Empty (train/val): {num_non_empty_clusters}, Max count: {count_max}, Min count: {count_min}")
