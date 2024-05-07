# %%
import os
os.environ["USL_MODE"] = "FINETUNE"


# %%

from utils import cfg, logger, print_b, print_y
import utils
import models.resnet_cifar as resnet_cifar
import models.resnet_cifar_cld as resnet_cifar_cld
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# %%
utils.init(default_config_file="configs/cifar10_usl_finetune.yaml")
logger.info(cfg)

# %%
print_b("Loading dataset")
assert cfg.DATASET.NAME in [
    "cifar10", "cifar100"], f"{cfg.DATASET.NAME} is not cifar10 or cifar100"
cifar100 = cfg.DATASET.NAME == "cifar100"
num_classes = 100 if cifar100 else 10

selected_inds = np.load(cfg.FINETUNE.LABELED_INDICES_PATH)

if len(selected_inds) <= 40:
    logger.info(f"Labeled Indices: {repr(selected_inds)}")

train_dataset_cifar, val_dataset = utils.train_dataset_cifar(
    transform_name=cfg.DATASET.TRANSFORM_NAME)

train_dataset_cifar.data = train_dataset_cifar.data[selected_inds]
select_targets = np.array(train_dataset_cifar.targets)[selected_inds]
train_dataset_cifar.targets = list(select_targets)
assert len(train_dataset_cifar.data) == len(train_dataset_cifar.targets)

# %%
print("Target dist:", np.unique(train_dataset_cifar.targets, return_counts=True))

if len(np.unique(train_dataset_cifar.targets)) != num_classes:
    logger.warning(f"WARNING: insufficient target: {len(np.unique(train_dataset_cifar.targets))} classes only")


if cfg.FINETUNE.REPEAT_DATA:
    train_dataset_cifar.data = np.vstack([train_dataset_cifar.data] * cfg.FINETUNE.REPEAT_DATA)
    train_dataset_cifar.targets = list(np.hstack([train_dataset_cifar.targets] * cfg.FINETUNE.REPEAT_DATA))

logger.info(f"Nunber of training samples: {len(train_dataset_cifar.targets)}")

train_dataloader = DataLoader(train_dataset_cifar, num_workers=cfg.DATALOADER.WORKERS,
                              batch_size=cfg.DATALOADER.BATCH_SIZE, pin_memory=True,
                              drop_last=True, shuffle=True)
val_dataloader = DataLoader(val_dataset, num_workers=cfg.DATALOADER.WORKERS,
                            batch_size=cfg.DATALOADER.BATCH_SIZE, pin_memory=True,
                            drop_last=False, shuffle=False)

train_targets = torch.tensor(train_dataset_cifar.targets)

val_targets = torch.tensor(val_dataset.targets)
train_targets.shape, val_targets.shape

# %%
print_b("Loading model")

checkpoint = torch.load(cfg.MODEL.PRETRAIN_PATH, map_location="cpu")

if cfg.MODEL.USE_CLD:
    state_dict = utils.single_model(checkpoint['model'])
else:
    state_dict = utils.single_model(checkpoint)
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

for k in list(state_dict.keys()):
    if k.startswith('linear') or k.startswith('fc') or k.startswith('groupDis'):
        del state_dict[k]

if cfg.MODEL.USE_CLD:
    model = resnet_cifar_cld.__dict__[cfg.MODEL.ARCH](
        low_dim=num_classes, pool_len=4, normlinear=False).cuda()
else:
    model = resnet_cifar.__dict__[cfg.MODEL.ARCH](num_classes=num_classes).cuda()

mismatch = model.load_state_dict(state_dict, strict=False)

logger.warning(
    f"Key mismatches: {mismatch} (extra contrastive keys are intended)")

# %%
print_b("Initializing optimizer")
torch.backends.cudnn.benchmark = True
if cfg.FINETUNE.FREEZE_BACKBONE:
    for name, param in model.named_parameters():
        if "linear" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

# params = [param for param in model.parameters() if param.requires_grad]
params = model.parameters()

optimizer = torch.optim.__dict__[cfg.OPTIMIZER.NAME](params,
                                                     lr=cfg.OPTIMIZER.LR,
                                                     momentum=cfg.OPTIMIZER.MOMENTUM,
                                                     nesterov=cfg.OPTIMIZER.NESTEROV,
                                                     weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY)

# %%
def train():
    model.train()

    for imgs, targets in tqdm(train_dataloader):
        imgs = imgs.cuda()
        targets = targets.cuda()

        pred = model(imgs)

        loss = F.cross_entropy(pred, targets)

        # print("Loss:", loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# %%
def eval():
    model.eval()

    preds = []
    all_targets = []

    confusion_matrix = torch.zeros((num_classes, num_classes))


    with torch.no_grad():
        for imgs, targets in tqdm(val_dataloader):
            imgs = imgs.cuda()

            pred = model(imgs)
            
            pred = pred.argmax(dim=1)

            for t, p in zip(targets.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            preds.append(pred)
            all_targets.append(targets)

    preds = torch.cat(preds, dim=0).cpu()
    all_targets = torch.cat(all_targets, dim=0)

    acc = torch.mean((preds == all_targets).float()).item()

    acc_per_class = confusion_matrix.diag()/confusion_matrix.sum(1)

    print("Eval Acc:", acc)
    print("Acc per class:", acc_per_class)

    return acc

# import ipdb; ipdb.set_trace()

# %%
for epoch in range(cfg.EPOCHS):
    print("Epoch", epoch + 1)
    train()
    acc = eval()

# %%
print("Final Acc:", acc)
