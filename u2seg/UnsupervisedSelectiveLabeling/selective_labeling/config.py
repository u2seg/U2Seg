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
# _C.DATALOADER.BATCH_SIZE = 128
_C.DATALOADER.BATCH_SIZE = 256
_C.DATALOADER.WORKERS = 2

_C.MODEL = CN()
_C.MODEL.ARCH = 'ResNet18'
_C.MODEL.PRETRAIN_PATH = ''
_C.MODEL.BACKBONE_DIM = 512

if mode == "USL":
    _C.USL = CN()
    _C.USL.NUM_SELECTED_SAMPLES = 40
    _C.USL.KNN_K = 400
    _C.USL.K_MEANS_NITERS = 200
    _C.USL.SEEDS = [1, 2, 3, 4, 5]

    _C.USL.REG = CN()
    _C.USL.REG.NITERS = 10
    # W is the lambda in math expressions
    _C.USL.REG.W = 0.5
    _C.USL.REG.MOMENTUM = 0.9
    _C.USL.REG.HORIZON_DIST = 4
    _C.USL.REG.ALPHA = 0.5
    _C.USL.REG.HORIZON_NUM = 128
    _C.USL.REG.EXCLUDE_SAME_CLUSTER = False

if mode == "RANDOM":
    _C.RANDOM = CN()
    _C.RANDOM.SEEDS = [1, 2, 3, 4, 5]
    _C.RANDOM.NUM_SELECTED_SAMPLES = 40

cfg = _C
