# %%
import os
os.environ["USL_MODE"] = "USLT_PRETRAIN"

from utils.uslt_utils import AverageMeter, ProgressMeter
from sklearn import metrics
from utils.dataloader import MultiEpochsDataLoader
from utils import uslt_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.resnet_cifar as resnet_cifar
import utils
from utils import cfg, logger, print_r, print_b, print_y, seed_everything

# %%

# %%
utils.init(default_config_file="configs/cifar10_usl-t_pretrain.yaml")
logger.info(cfg)
seed_everything(cfg.SEED)
if cfg.BACKUP_PYTHON_FILE and not cfg.EVAL_ONLY:
    import shutil
    shutil.copy(__file__, cfg.RUN_DIR)

# %%
print_b("Loading dataset")
assert cfg.DATASET.NAME in [
    "cifar10", "cifar100"], f"{cfg.DATASET.NAME} is not cifar10 or cifar100"
cifar100 = cfg.DATASET.NAME == "cifar100"
# num_classes = 100 if cifar100 else 10

train_dataset, val_dataset = utils.train_dataset_cifar(
    transform_name=cfg.DATASET.TRANSFORM_NAME, load_local_global_dataset=True)

train_dataloader = MultiEpochsDataLoader(train_dataset, num_workers=cfg.DATALOADER.WORKERS,
                              batch_size=cfg.DATALOADER.BATCH_SIZE, pin_memory=True, collate_fn=uslt_utils.collate_custom,
                              drop_last=True, shuffle=True)
val_dataloader = MultiEpochsDataLoader(val_dataset, num_workers=cfg.DATALOADER.WORKERS,
                            batch_size=cfg.DATALOADER.BATCH_SIZE, pin_memory=True, collate_fn=uslt_utils.collate_custom,
                            drop_last=False, shuffle=False)

val_targets = torch.tensor(val_dataset.targets)
val_targets.shape

# %%
print_b("Loading model")

checkpoint = torch.load(cfg.MODEL.PRETRAIN_PATH, map_location="cpu")

num_heads = 3
num_labeled = cfg.USLT_PRETRAIN.NUM_SELECTED_SAMPLES

backbone = resnet_cifar.__dict__[cfg.MODEL.ARCH]()
model = utils.ClusteringModel(backbone, nclusters=num_labeled, normed=False,
                        nheads=num_heads, backbone_dim=cfg.MODEL.BACKBONE_DIM).cuda()
mismatch = model.load_state_dict(utils.single_model(checkpoint), strict=False)

logger.warning(
    f"Key mismatches: {mismatch} (extra contrastive keys are intended)")

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
        if "head" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    params = [param for name, param in model.named_parameters()
              if "head" not in name]
else:
    params = model.parameters()

optimizer = torch.optim.__dict__[cfg.OPTIMIZER.NAME](params,
                                                     lr=cfg.OPTIMIZER.LR,
                                                     weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY)

# %%
print_b("Initializing losses")

criterion_local = utils.OursLossLocal(
    num_classes=num_labeled, num_heads=num_heads, adjustment_weight=cfg.USLT_PRETRAIN.ADJUSTMENT_WEIGHT, momentum=0.5, sharpen_temperature=cfg.USLT_PRETRAIN.SHARPEN_TEMPERATURE).cuda()
criterion_global = utils.OursLossGlobal(num_classes=num_labeled, num_heads=num_heads,
                                  threshold=cfg.USLT_PRETRAIN.CONFIDENCE_THRESHOLD, reweight=cfg.USLT_PRETRAIN.REWEIGHT, reweight_renorm=cfg.USLT_PRETRAIN.REWEIGHT_RENORM, mean_outside_mask=cfg.USLT_PRETRAIN.MEAN_OUTSIDE_MASK).cuda()
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

    for images, target in dataloader:
        images = images.cuda(non_blocking=True)
        output = model(images)

        for i, output_i in enumerate(output):
            predictions[i].append(torch.argmax(output_i, dim=1))
            probs[i].append(F.softmax(output_i, dim=1))
        targets.append(target)

    predictions = [torch.cat(pred_, dim=0).cpu() for pred_ in predictions]
    probs = [torch.cat(prob_, dim=0).cpu() for prob_ in probs]
    targets = torch.cat(targets, dim=0)

    out = [{'predictions': pred_, 'probabilities': prob_, 'targets': targets}
           for pred_, prob_ in zip(predictions, probs)]

    return out


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

    unique_clusters, cluster_sample_counts = torch.unique(predictions, return_counts=True)
    num_non_empty_clusters = len(unique_clusters)

    return macc, nmi, ari, num_non_empty_clusters, cluster_sample_counts.max().item(), cluster_sample_counts.min().item()


def ours_train(train_loader, model, ema_model, criterion, optimizer, epoch, update_head_only=False, cfg=None):
    """ 
    Train w/ Our Loss
    """
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

    for i, batch in enumerate(train_loader):
        # Forward pass
        batch_aug = batch['aug']

        # Global (k-Means) part
        images = batch_aug['image'].cuda(non_blocking=True)
        images_augmented = batch_aug['image_augmented'].cuda(non_blocking=True)

        with torch.no_grad():
            output = model(images)
        output_augmented = model(images_augmented)

        if epoch >= cfg.USLT_PRETRAIN.GLOBAL_START_EPOCH: # 0: start from epoch 0
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

        if i % 10 == 0:
            progress.display(i)


for epoch in range(cfg.EPOCHS):
    print_y('Epoch %d/%d' % (epoch+1, cfg.EPOCHS))

    # Train
    ours_train(train_dataloader, model, ema_model, criterion, optimizer,
               epoch, update_head_only=cfg.USLT_PRETRAIN.UPDATE_HEAD_ONLY, cfg=cfg)

    # Evaluate
    print_b('Make prediction on validation set ...')

    for ema_eval in [False, True]:
        if ema_eval:
            if ema_model is None:
                continue
            val_predictions = get_predictions(val_dataloader, ema_model.ema)
            ema_str = "EMA "
        else:
            val_predictions = get_predictions(val_dataloader, model)
            ema_str = ""
        for ind, val_predictions_head in enumerate(val_predictions):
            val_predictions, val_targets = val_predictions_head[
                "predictions"], val_predictions_head["targets"]
            macc, nmi, ari, num_non_empty_clusters, count_max, count_min = evaluate_predictions(val_predictions, val_targets)
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
