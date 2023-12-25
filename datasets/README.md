# Prepare Datasets for Training and Evaluating U2SEG

A dataset can be used by accessing [DatasetCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.DatasetCatalog)
for its data, or [MetadataCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.MetadataCatalog) for its metadata (class names, etc).
This document explains how to setup the builtin datasets so they can be used by the above APIs. [Use Custom Datasets](https://detectron2.readthedocs.io/tutorials/datasets.html) gives a deeper dive on how to use `DatasetCatalog` and `MetadataCatalog`,
and how to add new datasets to them.

U2Seg has builtin support for a few datasets. The datasets are assumed to exist in a directory specified by the environment variable `DETECTRON2_DATASETS`. Under this directory, detectron2 will look for datasets in the structure described below, if needed.
```
$DETECTRON2_DATASETS/datasets/datasets
  panoptic_anns/
  coco/
  ...
```

You can set the location for builtin datasets by `export DETECTRON2_DATASETS=/path/to/datasets`. If left unset, the default is `./datasets/datasets` relative to your current working directory.

All dataset annotation files should be converted to the COCO format.


## Expected dataset structure for [COCO](https://cocodataset.org/#download):
```
├── coco
│   ├── annotations
│   │   ├── instances_val2017.json
│   │   └── instances_train2017.json
|   |   └── ......
│   ├── train2017
│   │   └── 000000000009.jpg
|   |   └── 000000000025.jpg
|   |   └── ......
│   ├── val2017
│   │   └── 000000000139.jpg
|   |   └── 000000000285.jpg
|   |   └── ......
├── panoptic_anns
│   ├── panoptic_train2017.json
│   ├── panoptic_val2017.json
│   ├── panoptic_train2017
│   │   └── 000000000009.jpg
|   |   └── 000000000025.jpg
|   |   └── ......
│   ├── panoptic_val2017
│   │   └── 000000000139.jpg
|   |   └── 000000000285.jpg
|   |   └── ......
```
To generate the pseudo panoptic annotations for U2Seg training, we provide pre-generated anntation file at [click](https://drive.google.com/file/d/15t7pUWyLRijCiU79l4s10e8tWDgrt97h/view?usp=drive_link), then put it in the structure of:
```
$DETECTRON2_DATASETS/datasets
├── datasets
|   └── ......
├── prepare_ours
│   ├── u2seg_annotations
│   │   └── ins_annotations
|   |   └── semantic_annotations
|   |   └── panoptic_annotations (this will be generated using our code)

```
After you structure the file as above, you can generate the panoptic annotations by:

```
python ./datasets/prepare_ours/generate_pseudo_panoptic.py --class_num 800 --split train
```
This will automatically generate ```cocotrain_800``` and ```cocotrain_800.json``` at the ```/prepare_ours/panoptic_annotations```folder.

After that, run:
```
python ./datasets/prepare_ours/prepare_stuff_panoptic_fpn.py
```
to generate the stuff annotations.








NOTE: ALL DATASETS FOLLOW THEIR ORIGINAL LICENSES.
