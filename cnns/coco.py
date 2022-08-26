"""
implementation of coco dataset
"""
# pylint: disable=wrong-import-position
import os
import time
import logging
from pathlib import Path

import cv2
import torch
import numpy as np
import dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("coco")


class Coco(dataset.Dataset):
    """Coco dataset parser"""

    def __init__(
        self,
        data_path,
        image_list,
        name,
        use_cache=0,
        image_size=None,
        image_format="NHWC",
        pre_process=None,
        count=None,
        cache_dir=None,
        use_label_map=False,
    ):
        super().__init__()
        self.image_size = image_size
        self.image_ids = []
        self.image_sizes = []
        self.count = count  # None
        self.data_path = data_path
        self.pre_process = pre_process
        self.use_label_map = use_label_map  # True
        if not cache_dir:
            cache_dir = data_path
        self.cache_dir = os.path.join(cache_dir, "preprocessed", name, image_format, "val2017")
        # input images are in HWC
        self.need_transpose = bool(image_format == "NCHW")
        self.height, self.width, _ = self.image_size

        start = time.time()
        if image_list is None:
            yolov5_cache_path = os.path.join(data_path, "val2017.cache")
            cache = np.load(yolov5_cache_path, allow_pickle=True).item()
            n_found, n_missing, n_empty, n_corrupt, total = cache.pop('results')
            log.info(
                "Scanning '%s' images and labels... %d found, %d missing, \
                    %d empty, %d corrupt, %d total",
                yolov5_cache_path,
                n_found,
                n_missing,
                n_empty,
                n_corrupt,
                total,
            )
            for k in ('hash', 'version', 'msgs'):
                cache.pop(k)
            img_files = list(cache.keys())
            labels, shapes, _ = zip(*cache.values())
            self.label_list = list(labels)

            self.image_sizes = np.array(shapes, dtype=np.float64)

        for img in img_files:
            img_name = img.split("/")[-1]
            os.makedirs(os.path.dirname(os.path.join(self.cache_dir, img_name)), exist_ok=True)
            dst = os.path.join(self.cache_dir, img_name)
            if not os.path.exists(dst + ".npy"):
                # cache a preprocessed version of the image
                img_org = cv2.imread(img)
                processed = self.pre_process(img_org,
                                             dims=self.image_size,
                                             need_transpose=self.need_transpose)
                np.save(dst, processed)
            path = Path(img)
            image_id = int(path.stem) if path.stem.isnumeric() else path.stem

            self.image_list.append(img_name)
            self.image_ids.append(image_id)
            # limit the dataset if requested
            if self.count and len(self.image_list) >= self.count:
                break

        time_taken = time.time() - start
        if not self.image_list:
            log.error("no images in image list found")
            raise ValueError("no images in image list found")

        log.info(
            "loaded %d images, cache=%d, took=%.1fsec",
            len(self.image_list),
            use_cache,
            time_taken,
        )

        self.label_list = np.array(self.label_list)

    def get_item(self, index):
        """Get image by index in the list."""
        dst = os.path.join(self.cache_dir, self.image_list[index])
        img = np.load(dst + ".npy")
        label = torch.from_numpy(self.label_list[index])
        label[:, 1:] *= torch.Tensor([self.width, self.height, self.width, self.height])
        return img, label

    def get_item_loc(self, index):
        src = os.path.join(self.data_path, self.image_list[index])
        return src