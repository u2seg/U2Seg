import os
from yacs.config import CfgNode as CN

mode = os.environ["USL_MODE"]

_C = CN()
_C.RUN_NAME = ''
_C.SAVE_DIR = 'saved'
_C.SAVE_DIR_EXIST_OK = True
_C.RUN_DIR = ''
_C.SKIP_SAVE = False
# Include options in command line in RUN_NAME
_C.OPTS_IN_RUN_NAME = False

# Set to False to use the saved numpy files in RUN_DIR
_C.RECOMPUTE_ALL = True
# Set to True to recompute NUM_SELECTED_SAMPLES dependent steps such as k-Means and selection even if RECOMPUTE_ALL is False
_C.RECOMPUTE_NUM_DEP = True

_C.DATASET = CN()
_C.DATASET.NAME = ''
_C.DATASET.ROOT_DIR = '../data'
# CLD and FixMatch uses different normalization in transform
_C.DATASET.TRANSFORM_NAME = ''

_C.DATALOADER = CN()
_C.DATALOADER.BATCH_SIZE = 128
_C.DATALOADER.WORKERS = 8

_C.MODEL = CN()
_C.MODEL.ARCH = 'ResNet18'
_C.MODEL.PRETRAIN_PATH = ''
_C.MODEL.BACKBONE_DIM = 512
# USE_CLD: SimCLRv2-CLD, without USE_CLD: SimCLRv2 with SimCLR pretrained weights
_C.MODEL.USE_CLD = True

_C.OPTIMIZER = CN()
_C.OPTIMIZER.NAME = 'SGD'
_C.OPTIMIZER.LR = 0.01
_C.OPTIMIZER.MOMENTUM = 0.9
_C.OPTIMIZER.NESTEROV = True
_C.OPTIMIZER.WEIGHT_DECAY = 0.0001

_C.EPOCHS = 5

if mode == "FINETUNE":
    _C.FINETUNE = CN()
    _C.FINETUNE.LABELED_INDICES_PATH = ""
    _C.FINETUNE.FREEZE_BACKBONE = True
    _C.FINETUNE.REPEAT_DATA = 100


cfg = _C
