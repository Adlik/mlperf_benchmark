"""
Postprocessing yolov5 for coco dataset
"""
# pylint: disable=wrong-import-position
import json
import logging
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("coco-yolov5")

class PostProcessCocoPt:
    """
    Post processing for coco dataset
    """

    def __init__(self, score_threshold):
        self.results = []
        self.good = 0
        self.total = 0
        self.content_ids = []
        self.score_threshold = score_threshold

    def add_results(self, results):
        self.results.extend(results)

    def __call__(self, results, ids, expected=None):
        # results come as:
        # num_detections,detection_boxes,detection_scores,detection_classes
        processed_results = [ids]
        # detect_heads = DetectHead()
        # class_map = coco80_to_coco91_class()
        # out, _ = detect_heads(results)
        # out = non_max_suppression(out, self.score_threshold)

        # for idx, pred in enumerate(out):
        #     self.content_ids.append(ids[idx])
        #     processed_results.append([])

        #     box = xyxy2xywh(pred[:, :4])
        #     box[:, :2] -= box[:, 2:] / 2
        #     for p, b in zip(pred.tolist(), box.tolist()):
        #         processed_results[idx].append(
        #             [
        #                 float(ids[idx]),
        #                 b[0],
        #                 b[1],
        #                 b[2],
        #                 b[3],
        #                 class_map[int(p[5])],
        #                 round(p[4], 5),
        #             ]
        #         )
        return processed_results

    def start(self):
        self.results = []
        self.good = 0
        self.total = 0

    def finalize(self, result_dict, data=None):
        """Comupte accuracy of model on coco dataset"""
        result_dict["good"] += self.good
        result_dict["total"] += self.total

        if self.use_inv_map:
            # for pytorch
            label_map = {}
            with open(data.annotation_file, encoding="utf-8") as fin:
                annotations = json.load(fin)
            for cnt, cat in enumerate(annotations["categories"]):
                label_map[cat["id"]] = cnt + 1
            inv_map = {v: k for k, v in label_map.items()}

        detections = []
        image_indices = []
        for batch, results in enumerate(self.results):
            image_indices.append(self.content_ids[batch])
            for _, result in enumerate(results):
                detection = result
                # this is the index of the coco image
                image_idx = int(detection[0])
                if image_idx != self.content_ids[batch]:
                    # extra check to make sure it is consistent
                    log.error("image_idx missmatch, lg=%s / result=%s", image_idx,
                              self.content_ids[batch])
                # map the index to the coco image id
                detection[0] = data.image_ids[image_idx]
                height, width = data.image_sizes[image_idx]
                # box comes from model as: ymin, xmin, ymax, xmax
                ymin = detection[1] * height
                xmin = detection[2] * width
                ymax = detection[3] * height
                xmax = detection[4] * width
                # pycoco wants {imageID,x1,y1,w,h,score,class}
                detection[1] = xmin
                detection[2] = ymin
                detection[3] = xmax - xmin
                detection[4] = ymax - ymin
                if self.use_inv_map:
                    cat_id = inv_map.get(int(detection[6]), -1)
                    if cat_id == -1:
                        log.info("finalize can't map category %d", int(detection[6]))
                    detection[6] = cat_id
                detections.append(np.array(detection))

        # map indices to coco image id's
        image_ids = [data.image_ids[i] for i in image_indices]
        self.results = []
        cocogt = COCO(data.annotation_file)
        cocodt = cocogt.loadRes(np.array(detections))
        cocoeval = COCOeval(cocogt, cocodt, iouType='bbox')
        cocoeval.params.imgIds = image_ids
        cocoeval.evaluate()
        cocoeval.accumulate()
        cocoeval.summarize()
        result_dict["mAP"] = cocoeval.stats[0]
