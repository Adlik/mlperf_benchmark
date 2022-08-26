"""
Postprocessing maskrcnn for coco dataset
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from PIL import Image
import cv2
import torch
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_util

import torchvision.transforms.functional as F

import torch.nn.functional as FT
from collections import OrderedDict

from bounding_box import BoxList

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("coco-maskrcnn")

def expand_boxes(boxes, scale):
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    boxes_exp = torch.zeros_like(boxes)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half
    return boxes_exp


def expand_masks(mask, padding):
    N = mask.shape[0]
    M = mask.shape[-1]
    pad2 = 2 * padding
    scale = float(M + pad2) / M
    padded_mask = mask.new_zeros((N, 1, M + pad2, M + pad2))
    padded_mask[:, :, padding:-padding, padding:-padding] = mask
    return padded_mask, scale


def paste_mask_in_image(mask, box, im_h, im_w, thresh=0.5, padding=1):
    padded_mask, scale = expand_masks(mask[None], padding=padding)
    mask = padded_mask[0, 0]
    box = expand_boxes(box[None], scale)[0]
    box = box.to(dtype=torch.int32)

    TO_REMOVE = 1
    w = int(box[2] - box[0] + TO_REMOVE)
    h = int(box[3] - box[1] + TO_REMOVE)
    w = max(w, 1)
    h = max(h, 1)

    # Set shape to [batchxCxHxW]
    mask = mask.expand((1, 1, -1, -1))

    # Resize mask
    mask = mask.to(torch.float32)
    mask = FT.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
    mask = mask[0][0]

    if thresh >= 0:
        mask = mask > thresh
    else:
        # for visualization and debugging, we also
        # allow it to return an unmodified mask
        mask = (mask * 255).to(torch.uint8)

    im_mask = torch.zeros((im_h, im_w), dtype=torch.uint8)
    x_0 = max(box[0], 0)
    x_1 = min(box[2] + 1, im_w)
    y_0 = max(box[1], 0)
    y_1 = min(box[3] + 1, im_h)
   
    if y_0 <= y_1 and x_0 <= x_1:
        im_mask[y_0:y_1, x_0:x_1] = mask[(y_0 - box[1]):(y_1 - box[1]), (x_0 - box[0]):(x_1 - box[0])]

    return im_mask


class Masker(object):
    """
    Projects a set of masks in an image on the locations
    specified by the bounding boxes
    """

    def __init__(self, threshold=0.5, padding=1):
        self.threshold = threshold
        self.padding = padding

    def forward_single_image(self, masks, boxes):
        boxes = boxes.convert("xyxy")
        im_w, im_h = boxes.size
        res = [
            paste_mask_in_image(mask[0], box, im_h, im_w, self.threshold, self.padding)
            for mask, box in zip(masks, boxes.bbox)
        ]
        if len(res) > 0:
            res = torch.stack(res, dim=0)[:, None]
        else:
            res = masks.new_empty((0, 1, masks.shape[-2], masks.shape[-1]))
        return res

    def __call__(self, masks, boxes):
        if isinstance(boxes, BoxList):
            boxes = [boxes]

        # Make some sanity check
        assert len(boxes) == len(masks), "Masks and boxes should have the same length."

        # TODO:  Is this JIT compatible?
        # If not we should make it compatible.
        results = []
        for mask, box in zip(masks, boxes):
            assert mask.shape[0] == len(box), "Number of objects should be the same."
            result = self.forward_single_image(mask, box)
            results.append(result)
        return results



class COCOResults(object):
    METRICS = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "box_proposal": [
            "AR@100",
            "ARs@100",
            "ARm@100",
            "ARl@100",
            "AR@1000",
            "ARs@1000",
            "ARm@1000",
            "ARl@1000",
        ],
        "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
    }

    def __init__(self, iou_types):
        allowed_types = ("box_proposal", "bbox", "segm", "keypoints")
        assert all(iou_type in allowed_types for iou_type in iou_types)
        results = OrderedDict()
        for iou_type in iou_types:
            results[iou_type] = OrderedDict([(metric, -1)
                                             for metric in COCOResults.METRICS[iou_type]])
        self.results = results

    def update(self, coco_eval):
        if coco_eval is None:
            return

        assert isinstance(coco_eval, COCOeval)
        s = coco_eval.stats
        iou_type = coco_eval.params.iouType
        res = self.results[iou_type]
        metrics = COCOResults.METRICS[iou_type]
        for idx, metric in enumerate(metrics):
            res[metric] = s[idx]

    def __repr__(self):
        # TODO make it pretty
        return repr(self.results)


def remove_dup(l):
    seen = set()
    seen_add = seen.add
    return [x for x in l if not (x in seen or seen_add(x))]
#
# Post processing
#
class PostProcessCocoPt():
    """
    Post processing for coco dataset
    """

    def __init__(self, annotation_file):
        self.results = []
        self.good = 0
        self.total = 0
        self.coco = COCO(annotation_file)
        self.category2id = {v: i+1 for i, v in enumerate(self.coco.getCatIds())}
        self.id2category = {v: k for k, v in self.category2id.items()}
        self.coco_results = {}
        self.coco_bbox = []
        self.coco_segm = []
        self.id = list(sorted(self.coco.imgs.keys()))

    def add_results(self, results):
        self.results.extend(results)

    def prepare_for_coco_detection(self, predictions, ids):
        # assert isinstance(dataset, COCODataset)
        img_id = ids[0]
        img_id = self.id[img_id]
        img_info = self.coco.imgs[img_id]
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = BoxList(predictions[0], (800, 800), mode="xyxy")
        prediction = prediction.resize((image_width, image_height))
        prediction = prediction.convert("xywh")
        boxes = prediction.bbox.tolist()

        prediction.add_field("scores", predictions[2])   
        prediction.add_field("labels", predictions[1])     
        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()

        mapped_labels =[self.id2category[i] for i in labels]

        self.coco_bbox.extend(
            [
                {
                    "image_id": img_id,
                    "category_id": mapped_labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )

    def prepare_for_coco_segmentation(self, predictions, ids):

        masker = Masker(threshold=0.5, padding=1)
        img_id = ids[0]
        img_id = self.id[img_id]
        img_info = self.coco.imgs[img_id]
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = BoxList(predictions[0], (800, 800), mode="xyxy")
        prediction = prediction.resize((image_width, image_height))

        prediction.add_field("mask", predictions[3])    
        masks = prediction.get_field("mask")

        # Masker is necessary only if masks haven't been already resized.
        if list(masks.shape[-2:]) != [image_height, image_width]:
            masks = masker((torch.from_numpy(masks)).expand(1, -1, -1, -1, -1), prediction)
            masks = masks[0]

        prediction.add_field("scores", predictions[2])   
        prediction.add_field("labels", predictions[1])     
        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()

        rles = [
            mask_util.encode(np.array(mask[0, :, :, np.newaxis], order="F"))[0]
            for mask in masks
        ]


        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")

        mapped_labels = [self.id2category[i] for i in labels]
 
        self.coco_segm.extend([{
            "image_id": img_id,
            "category_id": mapped_labels[k],
            "segmentation": rle,
            "score": scores[k],
        } for k, rle in enumerate(rles)])


    def __call__(self, results, ids, expected=None):
        log.info("Preparing results for COCO format")
        processed_results = [ids]
        self.prepare_for_coco_detection(results, ids)
        self.prepare_for_coco_segmentation(results, ids)    
        return processed_results

    def start(self):
        self.results = []
        self.good = 0
        self.total = 0

    def evaluate_predictions_on_coco(self, coco_gt, coco_results, json_result_file, iou_type="bbox"):
        set_of_json = remove_dup([json.dumps(d) for d in coco_results])
        unique_list = [json.loads(s) for s in set_of_json]

        with open(json_result_file, "w") as f:
            json.dump(unique_list, f)

        coco_dt = coco_gt.loadRes(str(json_result_file)) if self.coco_results else COCO()

        coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval

    def finalize(self, result_dict, data=None, output_dir=None, iou_types=("bbox", "segm")):
        """Comupte accuracy of model on coco dataset"""
        log.info("Evaluating predictions")
        if "bbox" in iou_types:
            self.coco_results["bbox"] = self.coco_bbox
        if "segm" in iou_types:
            self.coco_results["segm"] = self.coco_segm
        results = COCOResults(iou_types)

        for iou_type in iou_types:
            file_path = os.path.join("/tmp", iou_type + ".json")
            res = self.evaluate_predictions_on_coco(self.coco, self.coco_results[iou_type], file_path, iou_type)
            results.update(res)
        log.info(results)