# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
from typing import Optional, Union
import pycocotools.mask as mask_util
import torch
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager

from .evaluator import DatasetEvaluator

_CV2_IMPORTED = True
try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    _CV2_IMPORTED = False


def load_image_into_numpy_array(
    filename: str,
    copy: bool = False,
    dtype: Optional[Union[np.dtype, str]] = None,
) -> np.ndarray:
    with PathManager.open(filename, "rb") as f:
        array = np.array(Image.open(f), copy=copy, dtype=dtype)
    return array


class SemSegEvaluator(DatasetEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        *,
        sem_seg_loading_fn=load_image_into_numpy_array,
        num_classes=None,
        ignore_label=None,
        mode='hungarain_matching'
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
            sem_seg_loading_fn: function to read sem seg file and load into numpy array.
                Default provided, but projects can customize.
            num_classes, ignore_label: deprecated argument
        """
        self._logger = logging.getLogger(__name__)
        if num_classes is not None:
            self._logger.warn(
                "SemSegEvaluator(num_classes) is deprecated! It should be obtained from metadata."
            )
        if ignore_label is not None:
            self._logger.warn(
                "SemSegEvaluator(ignore_label) is deprecated! It should be obtained from metadata."
            )
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")

        self.input_file_to_gt_file = {
            dataset_record["file_name"]: dataset_record["sem_seg_file_name"]
            for dataset_record in DatasetCatalog.get(dataset_name)
        }

        meta = MetadataCatalog.get(dataset_name)
        # Dict that maps contiguous training ids to COCO category ids
        try:
            c2d = meta.stuff_dataset_id_to_contiguous_id
            self._contiguous_id_to_dataset_id = {v: k for k, v in c2d.items()}
        except AttributeError:
            self._contiguous_id_to_dataset_id = None

        self.supercateg_map = {1: 'textile', 2: 'building', 3: 'raw-material', 4: 'furniture-stuff', 5: 'floor', 6: 'plant',
                    7: 'food-stuff', 8: 'ground', 9: 'structural', 10: 'water', 11: 'wall', 12: 'window', 13: 'ceiling',
                    14: 'sky', 15: 'solid'}
        # self._class_names = meta.stuff_classes
        self._class_names = []
        for i in range(16):
            if i==0:
                self._class_names.append('things')
            else:
                self._class_names.append(self.supercateg_map[i])

        self.sem_seg_loading_fn = sem_seg_loading_fn
        self._num_classes = len(meta.stuff_classes)
        if num_classes is not None:
            assert self._num_classes == num_classes, f"{self._num_classes} != {num_classes}"
        self._ignore_label = ignore_label if ignore_label is not None else meta.ignore_label

        # This is because cv2.erode did not work for int datatype. Only works for uint8.
        self._compute_boundary_iou = True
        if not _CV2_IMPORTED:
            self._compute_boundary_iou = False
            self._logger.warn(
                """Boundary IoU calculation requires OpenCV. B-IoU metrics are
                not going to be computed because OpenCV is not available to import."""
            )
        if self._num_classes >= np.iinfo(np.uint8).max:
            self._compute_boundary_iou = False
            self._logger.warn(
                f"""SemSegEvaluator(num_classes) is more than supported value for Boundary IoU calculation!
                B-IoU metrics are not going to be computed. Max allowed value (exclusive)
                for num_classes for calculating Boundary IoU is {np.iinfo(np.uint8).max}.
                The number of classes of dataset {self._dataset_name} is {self._num_classes}"""
            )

        self.pseudo_gt_cate = []
        self.pred_det_cate = []
        self.pred_det_conf_score = []
        # self._num_classes = 16#54  # this number should be stuff clas + 1
        self.mode = mode
        self.hungarain_matching_save_path = './hungarian_matching/semantic_mapping.json'
        self._num_classes = 16
        # if self.mode == 'eval':
        #     self._num_classes = 16
        # else:
        #     self._num_classes = 54



    def reset(self):
        self._conf_matrix = np.zeros((self._num_classes + 1, self._num_classes + 1), dtype=np.int64)
        self._b_conf_matrix = np.zeros(
            (self._num_classes + 1, self._num_classes + 1), dtype=np.int64
        )
        self._predictions = []

    def hungarain_matching(self, all_preds, all_targets, num_labeled, num_classes):
        n_nomapping = 0
        mapping_dict = {}
        for i in range(1, num_labeled + 1):
            labeled_i_mask = all_preds == i
            labeled_i_target =np.bincount(all_targets[labeled_i_mask], minlength=num_classes)
            if np.sum(labeled_i_target) == 0:
                mapping_dict[i] = -1
                n_nomapping += 1
            else:
                mapping_dict[i] =np.argmax(labeled_i_target).item()
        mapping_dict[0] = 0
        print(n_nomapping)
        return mapping_dict

    def transfer(self, gt):
        """
        this function is to trnasfer the category to supercategory for evaluations
        :param gt: ground truth for a specific image
        :return: ground truth but shown as supercategories, for coco it should be 16 unique categories (15 supercate + 0 for things + 255 for ignore)
        """

        categories = [{'supercategory': 'person', 'isthing': 1, 'id': 1, 'name': 'person'}, {'supercategory': 'vehicle', 'isthing': 1, 'id': 2, 'name': 'bicycle'}, {'supercategory': 'vehicle', 'isthing': 1, 'id': 3, 'name': 'car'}, {'supercategory': 'vehicle', 'isthing': 1, 'id': 4, 'name': 'motorcycle'}, {'supercategory': 'vehicle', 'isthing': 1, 'id': 5, 'name': 'airplane'}, {'supercategory': 'vehicle', 'isthing': 1, 'id': 6, 'name': 'bus'}, {'supercategory': 'vehicle', 'isthing': 1, 'id': 7, 'name': 'train'}, {'supercategory': 'vehicle', 'isthing': 1, 'id': 8, 'name': 'truck'}, {'supercategory': 'vehicle', 'isthing': 1, 'id': 9, 'name': 'boat'}, {'supercategory': 'outdoor', 'isthing': 1, 'id': 10, 'name': 'traffic light'}, {'supercategory': 'outdoor', 'isthing': 1, 'id': 11, 'name': 'fire hydrant'}, {'supercategory': 'outdoor', 'isthing': 1, 'id': 13, 'name': 'stop sign'}, {'supercategory': 'outdoor', 'isthing': 1, 'id': 14, 'name': 'parking meter'}, {'supercategory': 'outdoor', 'isthing': 1, 'id': 15, 'name': 'bench'}, {'supercategory': 'animal', 'isthing': 1, 'id': 16, 'name': 'bird'}, {'supercategory': 'animal', 'isthing': 1, 'id': 17, 'name': 'cat'}, {'supercategory': 'animal', 'isthing': 1, 'id': 18, 'name': 'dog'}, {'supercategory': 'animal', 'isthing': 1, 'id': 19, 'name': 'horse'}, {'supercategory': 'animal', 'isthing': 1, 'id': 20, 'name': 'sheep'}, {'supercategory': 'animal', 'isthing': 1, 'id': 21, 'name': 'cow'}, {'supercategory': 'animal', 'isthing': 1, 'id': 22, 'name': 'elephant'}, {'supercategory': 'animal', 'isthing': 1, 'id': 23, 'name': 'bear'}, {'supercategory': 'animal', 'isthing': 1, 'id': 24, 'name': 'zebra'}, {'supercategory': 'animal', 'isthing': 1, 'id': 25, 'name': 'giraffe'}, {'supercategory': 'accessory', 'isthing': 1, 'id': 27, 'name': 'backpack'}, {'supercategory': 'accessory', 'isthing': 1, 'id': 28, 'name': 'umbrella'}, {'supercategory': 'accessory', 'isthing': 1, 'id': 31, 'name': 'handbag'}, {'supercategory': 'accessory', 'isthing': 1, 'id': 32, 'name': 'tie'}, {'supercategory': 'accessory', 'isthing': 1, 'id': 33, 'name': 'suitcase'}, {'supercategory': 'sports', 'isthing': 1, 'id': 34, 'name': 'frisbee'}, {'supercategory': 'sports', 'isthing': 1, 'id': 35, 'name': 'skis'}, {'supercategory': 'sports', 'isthing': 1, 'id': 36, 'name': 'snowboard'}, {'supercategory': 'sports', 'isthing': 1, 'id': 37, 'name': 'sports ball'}, {'supercategory': 'sports', 'isthing': 1, 'id': 38, 'name': 'kite'}, {'supercategory': 'sports', 'isthing': 1, 'id': 39, 'name': 'baseball bat'}, {'supercategory': 'sports', 'isthing': 1, 'id': 40, 'name': 'baseball glove'}, {'supercategory': 'sports', 'isthing': 1, 'id': 41, 'name': 'skateboard'}, {'supercategory': 'sports', 'isthing': 1, 'id': 42, 'name': 'surfboard'}, {'supercategory': 'sports', 'isthing': 1, 'id': 43, 'name': 'tennis racket'}, {'supercategory': 'kitchen', 'isthing': 1, 'id': 44, 'name': 'bottle'}, {'supercategory': 'kitchen', 'isthing': 1, 'id': 46, 'name': 'wine glass'}, {'supercategory': 'kitchen', 'isthing': 1, 'id': 47, 'name': 'cup'}, {'supercategory': 'kitchen', 'isthing': 1, 'id': 48, 'name': 'fork'}, {'supercategory': 'kitchen', 'isthing': 1, 'id': 49, 'name': 'knife'}, {'supercategory': 'kitchen', 'isthing': 1, 'id': 50, 'name': 'spoon'}, {'supercategory': 'kitchen', 'isthing': 1, 'id': 51, 'name': 'bowl'}, {'supercategory': 'food', 'isthing': 1, 'id': 52, 'name': 'banana'}, {'supercategory': 'food', 'isthing': 1, 'id': 53, 'name': 'apple'}, {'supercategory': 'food', 'isthing': 1, 'id': 54, 'name': 'sandwich'}, {'supercategory': 'food', 'isthing': 1, 'id': 55, 'name': 'orange'}, {'supercategory': 'food', 'isthing': 1, 'id': 56, 'name': 'broccoli'}, {'supercategory': 'food', 'isthing': 1, 'id': 57, 'name': 'carrot'}, {'supercategory': 'food', 'isthing': 1, 'id': 58, 'name': 'hot dog'}, {'supercategory': 'food', 'isthing': 1, 'id': 59, 'name': 'pizza'}, {'supercategory': 'food', 'isthing': 1, 'id': 60, 'name': 'donut'}, {'supercategory': 'food', 'isthing': 1, 'id': 61, 'name': 'cake'}, {'supercategory': 'furniture', 'isthing': 1, 'id': 62, 'name': 'chair'}, {'supercategory': 'furniture', 'isthing': 1, 'id': 63, 'name': 'couch'}, {'supercategory': 'furniture', 'isthing': 1, 'id': 64, 'name': 'potted plant'}, {'supercategory': 'furniture', 'isthing': 1, 'id': 65, 'name': 'bed'}, {'supercategory': 'furniture', 'isthing': 1, 'id': 67, 'name': 'dining table'}, {'supercategory': 'furniture', 'isthing': 1, 'id': 70, 'name': 'toilet'}, {'supercategory': 'electronic', 'isthing': 1, 'id': 72, 'name': 'tv'}, {'supercategory': 'electronic', 'isthing': 1, 'id': 73, 'name': 'laptop'}, {'supercategory': 'electronic', 'isthing': 1, 'id': 74, 'name': 'mouse'}, {'supercategory': 'electronic', 'isthing': 1, 'id': 75, 'name': 'remote'}, {'supercategory': 'electronic', 'isthing': 1, 'id': 76, 'name': 'keyboard'}, {'supercategory': 'electronic', 'isthing': 1, 'id': 77, 'name': 'cell phone'}, {'supercategory': 'appliance', 'isthing': 1, 'id': 78, 'name': 'microwave'}, {'supercategory': 'appliance', 'isthing': 1, 'id': 79, 'name': 'oven'}, {'supercategory': 'appliance', 'isthing': 1, 'id': 80, 'name': 'toaster'}, {'supercategory': 'appliance', 'isthing': 1, 'id': 81, 'name': 'sink'}, {'supercategory': 'appliance', 'isthing': 1, 'id': 82, 'name': 'refrigerator'}, {'supercategory': 'indoor', 'isthing': 1, 'id': 84, 'name': 'book'}, {'supercategory': 'indoor', 'isthing': 1, 'id': 85, 'name': 'clock'}, {'supercategory': 'indoor', 'isthing': 1, 'id': 86, 'name': 'vase'}, {'supercategory': 'indoor', 'isthing': 1, 'id': 87, 'name': 'scissors'}, {'supercategory': 'indoor', 'isthing': 1, 'id': 88, 'name': 'teddy bear'}, {'supercategory': 'indoor', 'isthing': 1, 'id': 89, 'name': 'hair drier'}, {'supercategory': 'indoor', 'isthing': 1, 'id': 90, 'name': 'toothbrush'}, {'supercategory': 'textile', 'isthing': 0, 'id': 92, 'name': 'banner'}, {'supercategory': 'textile', 'isthing': 0, 'id': 93, 'name': 'blanket'}, {'supercategory': 'building', 'isthing': 0, 'id': 95, 'name': 'bridge'}, {'supercategory': 'raw-material', 'isthing': 0, 'id': 100, 'name': 'cardboard'}, {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 107, 'name': 'counter'}, {'supercategory': 'textile', 'isthing': 0, 'id': 109, 'name': 'curtain'}, {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 112, 'name': 'door-stuff'}, {'supercategory': 'floor', 'isthing': 0, 'id': 118, 'name': 'floor-wood'}, {'supercategory': 'plant', 'isthing': 0, 'id': 119, 'name': 'flower'}, {'supercategory': 'food-stuff', 'isthing': 0, 'id': 122, 'name': 'fruit'}, {'supercategory': 'ground', 'isthing': 0, 'id': 125, 'name': 'gravel'}, {'supercategory': 'building', 'isthing': 0, 'id': 128, 'name': 'house'}, {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 130, 'name': 'light'}, {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 133, 'name': 'mirror-stuff'}, {'supercategory': 'structural', 'isthing': 0, 'id': 138, 'name': 'net'}, {'supercategory': 'textile', 'isthing': 0, 'id': 141, 'name': 'pillow'}, {'supercategory': 'ground', 'isthing': 0, 'id': 144, 'name': 'platform'}, {'supercategory': 'ground', 'isthing': 0, 'id': 145, 'name': 'playingfield'}, {'supercategory': 'ground', 'isthing': 0, 'id': 147, 'name': 'railroad'}, {'supercategory': 'water', 'isthing': 0, 'id': 148, 'name': 'river'}, {'supercategory': 'ground', 'isthing': 0, 'id': 149, 'name': 'road'}, {'supercategory': 'building', 'isthing': 0, 'id': 151, 'name': 'roof'}, {'supercategory': 'ground', 'isthing': 0, 'id': 154, 'name': 'sand'}, {'supercategory': 'water', 'isthing': 0, 'id': 155, 'name': 'sea'}, {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 156, 'name': 'shelf'}, {'supercategory': 'ground', 'isthing': 0, 'id': 159, 'name': 'snow'}, {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 161, 'name': 'stairs'}, {'supercategory': 'building', 'isthing': 0, 'id': 166, 'name': 'tent'}, {'supercategory': 'textile', 'isthing': 0, 'id': 168, 'name': 'towel'}, {'supercategory': 'wall', 'isthing': 0, 'id': 171, 'name': 'wall-brick'}, {'supercategory': 'wall', 'isthing': 0, 'id': 175, 'name': 'wall-stone'}, {'supercategory': 'wall', 'isthing': 0, 'id': 176, 'name': 'wall-tile'}, {'supercategory': 'wall', 'isthing': 0, 'id': 177, 'name': 'wall-wood'}, {'supercategory': 'water', 'isthing': 0, 'id': 178, 'name': 'water-other'}, {'supercategory': 'window', 'isthing': 0, 'id': 180, 'name': 'window-blind'}, {'supercategory': 'window', 'isthing': 0, 'id': 181, 'name': 'window-other'}, {'supercategory': 'plant', 'isthing': 0, 'id': 184, 'name': 'tree-merged'}, {'supercategory': 'structural', 'isthing': 0, 'id': 185, 'name': 'fence-merged'}, {'supercategory': 'ceiling', 'isthing': 0, 'id': 186, 'name': 'ceiling-merged'}, {'supercategory': 'sky', 'isthing': 0, 'id': 187, 'name': 'sky-other-merged'}, {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 188, 'name': 'cabinet-merged'}, {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 189, 'name': 'table-merged'}, {'supercategory': 'floor', 'isthing': 0, 'id': 190, 'name': 'floor-other-merged'}, {'supercategory': 'ground', 'isthing': 0, 'id': 191, 'name': 'pavement-merged'}, {'supercategory': 'solid', 'isthing': 0, 'id': 192, 'name': 'mountain-merged'}, {'supercategory': 'plant', 'isthing': 0, 'id': 193, 'name': 'grass-merged'}, {'supercategory': 'ground', 'isthing': 0, 'id': 194, 'name': 'dirt-merged'}, {'supercategory': 'raw-material', 'isthing': 0, 'id': 195, 'name': 'paper-merged'}, {'supercategory': 'food-stuff', 'isthing': 0, 'id': 196, 'name': 'food-other-merged'}, {'supercategory': 'building', 'isthing': 0, 'id': 197, 'name': 'building-other-merged'}, {'supercategory': 'solid', 'isthing': 0, 'id': 198, 'name': 'rock-merged'}, {'supercategory': 'wall', 'isthing': 0, 'id': 199, 'name': 'wall-other-merged'}, {'supercategory': 'textile', 'isthing': 0, 'id': 200, 'name': 'rug-merged'}]

        stuff_ids = [k["id"] for k in categories if k["isthing"] == 0]
        thing_ids = [k["id"] for k in categories if k["isthing"] == 1]

        id_map = {}  # map the 133 continuous coco id to 54 id (53 semnatic id + 0 for things)
        assert len(stuff_ids) <= 254
        for i, stuff_id in enumerate(stuff_ids):
            id_map[stuff_id] = i + 1
        for thing_id in thing_ids:
            id_map[thing_id] = 0
        id_map[0] = 255 # add an ignore id 0 in addition to 133 that point to 255

        id_map_ = {} # this is the reverse of the id_map
        for k in id_map:
            if id_map[k] == 0 or id_map[k]==255:
                continue
            id_map_[id_map[k]] = k


        map = {92: 1, 93: 1, 95: 2, 100: 3, 107: 4, 109: 1, 112: 4, 118: 5, 119: 6, 122: 7, 125: 8, 128: 2, 130: 4,
               133: 4, 138: 9, 141: 1, 144: 8, 145: 8, 147: 8, 148: 10, 149: 8, 151: 2, 154: 8, 155: 10, 156: 4, 159: 8,
               161: 4, 166: 2, 168: 1, 171: 11, 175: 11, 176: 11, 177: 11, 178: 10, 180: 12, 181: 12, 184: 6, 185: 9,
               186: 13, 187: 14, 188: 4, 189: 4, 190: 5, 191: 8, 192: 15, 193: 6, 194: 8, 195: 3, 196: 7, 197: 2,
               198: 15, 199: 11, 200: 1} # this is to map 53 semantic id to their supercategory

        for _gt in np.unique(gt):
            if _gt == 0 or _gt == 255:
                continue
            else:
                dataset_id = id_map_[_gt]
                supercate_id = map[dataset_id]
                gt[gt == _gt] = supercate_id
        return gt # the return is supercategoty with 0 for things, 255 for ignore

    def do_hangarain_mapping(self, coco_results, gt):
        """
        this function save the current batch prediction and the ground truth for final hungarian matching
        :param coco_results: the predictions
        :param gt: the ground truth
        :return:
        """
        pred_num_mask = np.unique(coco_results)
        gt_num_mask = np.unique(gt)

        for pred in pred_num_mask:
            if pred == 0:
                continue
            mask_pred = (coco_results == pred)
            for _gt in gt_num_mask:
                if _gt == 0 or _gt == 16:
                    continue
                mask_gt = (gt == _gt)

                iou = np.sum((mask_pred * mask_gt)) / np.sum((mask_pred + mask_gt))

                if iou > 0.15:  # TODO: find that thresh, the results somehow is sensitive to this threshold
                    self.pseudo_gt_cate.append(_gt)
                    self.pred_det_cate.append(pred)
                    continue

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output in zip(inputs, outputs):
            output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
            pred = np.array(output, dtype=int)
            gt_filename = self.input_file_to_gt_file[input["file_name"]]
            gt = self.sem_seg_loading_fn(gt_filename, dtype=int)


            if self.mode == 'hungarian_matching':
                _gt = self.transfer(gt)
                _gt[_gt == self._ignore_label] = self._num_classes

                # here for hungarian_matching
                self.do_hangarain_mapping(coco_results=pred, gt=_gt)

            elif self.mode == 'eval':
                # transfer to supercategory
                gt = self.transfer(gt)
                gt[gt == self._ignore_label] = self._num_classes
                mapping_dict = json.load(open(self.hungarain_matching_save_path))
                for cls in mapping_dict:
                    if mapping_dict[cls] == -1:
                        pred[pred == int(cls)] = self._num_classes
                    else:
                        pred[pred == int(cls)] = mapping_dict[cls]

                self._conf_matrix += np.bincount(
                    (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                    minlength=self._conf_matrix.size,
                ).reshape(self._conf_matrix.shape)

                if self._compute_boundary_iou:
                    b_gt = self._mask_to_boundary(gt.astype(np.uint8))
                    b_pred = self._mask_to_boundary(pred.astype(np.uint8))

                    self._b_conf_matrix += np.bincount(
                        (self._num_classes + 1) * b_pred.reshape(-1) + b_gt.reshape(-1),
                        minlength=self._conf_matrix.size,
                    ).reshape(self._conf_matrix.shape)

                # self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """

        if self.mode == 'hungarian_matching':
            mapping_dict = self.hungarain_matching(all_preds=np.array(self.pred_det_cate), all_targets=np.array(self.pseudo_gt_cate),
                                                   num_classes=15, num_labeled=27)
            print(mapping_dict)
            # save the mapping dict
            save_path = self.hungarain_matching_save_path
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(mapping_dict, f, ensure_ascii=False)

            results = OrderedDict({"sem_seg": None})
            self._logger.info(results)


        elif self.mode == 'eval':
            if self._distributed:
                synchronize()
                conf_matrix_list = all_gather(self._conf_matrix)
                b_conf_matrix_list = all_gather(self._b_conf_matrix)
                self._predictions = all_gather(self._predictions)
                self._predictions = list(itertools.chain(*self._predictions))
                if not is_main_process():
                    return

                self._conf_matrix = np.zeros_like(self._conf_matrix)
                for conf_matrix in conf_matrix_list:
                    self._conf_matrix += conf_matrix

                self._b_conf_matrix = np.zeros_like(self._b_conf_matrix)
                for b_conf_matrix in b_conf_matrix_list:
                    self._b_conf_matrix += b_conf_matrix

            # if self._output_dir:
            #     PathManager.mkdirs(self._output_dir)
            #     file_path = os.path.join(self._output_dir, "sem_seg_predictions.json")
            #     with PathManager.open(file_path, "w") as f:
            #         f.write(json.dumps(self._predictions))

            acc = np.full(self._num_classes, np.nan, dtype=float)
            iou = np.full(self._num_classes, np.nan, dtype=float)
            tp = self._conf_matrix.diagonal()[:-1].astype(float)
            pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(float)
            class_weights = pos_gt / np.sum(pos_gt)
            pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(float)
            acc_valid = pos_gt > 0
            acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
            union = pos_gt + pos_pred - tp
            iou_valid = np.logical_and(acc_valid, union > 0)
            iou[iou_valid] = tp[iou_valid] / union[iou_valid]
            macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
            miou = np.sum(iou[iou_valid]) / np.sum(iou_valid)
            fiou = np.sum(iou[iou_valid] * class_weights[iou_valid])
            pacc = np.sum(tp) / np.sum(pos_gt)

            if self._compute_boundary_iou:
                b_iou = np.full(self._num_classes, np.nan, dtype=float)
                b_tp = self._b_conf_matrix.diagonal()[:-1].astype(float)
                b_pos_gt = np.sum(self._b_conf_matrix[:-1, :-1], axis=0).astype(float)
                b_pos_pred = np.sum(self._b_conf_matrix[:-1, :-1], axis=1).astype(float)
                b_union = b_pos_gt + b_pos_pred - b_tp
                b_iou_valid = b_union > 0
                b_iou[b_iou_valid] = b_tp[b_iou_valid] / b_union[b_iou_valid]

            res = {}
            res["mIoU"] = 100 * miou
            res["fwIoU"] = 100 * fiou
            for i, name in enumerate(self._class_names):
                res[f"IoU-{name}"] = 100 * iou[i]
                if self._compute_boundary_iou:
                    res[f"BoundaryIoU-{name}"] = 100 * b_iou[i]
                    res[f"min(IoU, B-Iou)-{name}"] = 100 * min(iou[i], b_iou[i])
            res["mACC"] = 100 * macc
            res["pACC"] = 100 * pacc
            for i, name in enumerate(self._class_names):
                res[f"ACC-{name}"] = 100 * acc[i]

            if self._output_dir:
                file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
                with PathManager.open(file_path, "wb") as f:
                    torch.save(res, f)
            results = OrderedDict({"sem_seg": res})
            self._logger.info(results)
        return results

    def encode_json_sem_seg(self, sem_seg, input_file_name):
        """
        Convert semantic segmentation to COCO stuff format with segments encoded as RLEs.
        See http://cocodataset.org/#format-results
        """
        json_list = []
        for label in np.unique(sem_seg):
            if self._contiguous_id_to_dataset_id is not None:
                assert (
                    label in self._contiguous_id_to_dataset_id
                ), "Label {} is not in the metadata info for {}".format(label, self._dataset_name)
                dataset_id = self._contiguous_id_to_dataset_id[label]
            else:
                dataset_id = int(label)
            mask = (sem_seg == label).astype(np.uint8)
            mask_rle = mask_util.encode(np.array(mask[:, :, None], order="F"))[0]
            mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
            json_list.append(
                {"file_name": input_file_name, "category_id": dataset_id, "segmentation": mask_rle}
            )
        return json_list

    def _mask_to_boundary(self, mask: np.ndarray, dilation_ratio=0.02):
        assert mask.ndim == 2, "mask_to_boundary expects a 2-dimensional image"
        h, w = mask.shape
        diag_len = np.sqrt(h**2 + w**2)
        dilation = max(1, int(round(dilation_ratio * diag_len)))
        kernel = np.ones((3, 3), dtype=np.uint8)

        padded_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        eroded_mask_with_padding = cv2.erode(padded_mask, kernel, iterations=dilation)
        eroded_mask = eroded_mask_with_padding[1:-1, 1:-1]
        boundary = mask - eroded_mask
        return boundary
