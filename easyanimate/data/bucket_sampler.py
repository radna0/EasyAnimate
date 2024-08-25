# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import (
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Sized,
    TypeVar,
    Union,
)

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import BatchSampler, Dataset, Sampler

ASPECT_RATIO_1280 = {
    "0.25": [640.0, 2560.0],
    "0.26": [640.0, 2482.7],
    "0.27": [640.0, 2400.0],
    "0.28": [640.0, 2317.3],
    "0.32": [720.0, 2250.7],
    "0.33": [720.0, 2160.0],
    "0.35": [720.0, 2080.0],
    "0.4": [800.0, 2000.0],
    "0.42": [800.0, 1920.0],
    "0.48": [880.0, 1840.0],
    "0.5": [880.0, 1760.0],
    "0.52": [880.0, 1680.0],
    "0.57": [960.0, 1680.0],
    "0.6": [960.0, 1600.0],
    "0.68": [1045.3, 1536.0],
    "0.72": [1045.3, 1442.7],
    "0.78": [1122.7, 1442.7],
    "0.82": [1122.7, 1360.0],
    "0.88": [1200.0, 1360.0],
    "0.94": [1200.0, 1280.0],
    "1.0": [1280.0, 1280.0],
    "1.07": [1280.0, 1200.0],
    "1.13": [1360.0, 1200.0],
    "1.21": [1360.0, 1122.7],
    "1.29": [1442.7, 1122.7],
    "1.38": [1442.7, 1045.3],
    "1.46": [1525.3, 1045.3],
    "1.67": [1600.0, 960.0],
    "1.75": [1680.0, 960.0],
    "2.0": [1760.0, 880.0],
    "2.09": [1840.0, 880.0],
    "2.4": [1920.0, 800.0],
    "2.5": [2000.0, 800.0],
    "2.89": [2080.0, 720.0],
    "3.0": [2160.0, 720.0],
    "3.11": [2250.7, 720.0],
    "3.62": [2317.3, 640.0],
    "3.75": [2400.0, 640.0],
    "3.88": [2482.7, 640.0],
    "4.0": [2560.0, 640.0],
}

ASPECT_RATIO_RANDOM_CROP_1280 = {
    "0.42": [800.0, 1920.0],
    "0.5": [880.0, 1760.0],
    "0.57": [960.0, 1680.0],
    "0.68": [1045.3, 1536.0],
    "0.78": [1122.7, 1442.7],
    "0.88": [1200.0, 1360.0],
    "0.94": [1200.0, 1280.0],
    "1.0": [1280.0, 1280.0],
    "1.07": [1280.0, 1200.0],
    "1.13": [1360.0, 1200.0],
    "1.29": [1442.7, 1122.7],
    "1.46": [1525.3, 1045.3],
    "1.75": [1680.0, 960.0],
    "2.0": [1760.0, 880.0],
    "2.4": [1920.0, 800.0],
}



ASPECT_RATIO_1024 = {
    "0.25": [512.0, 2048.0],
    "0.26": [512.0, 1988.8],
    "0.27": [512.0, 1920.0],
    "0.28": [512.0, 1852.3],
    "0.32": [576.0, 1797.3],
    "0.33": [576.0, 1728.0],
    "0.35": [576.0, 1664.0],
    "0.4": [640.0, 1600.0],
    "0.42": [640.0, 1536.0],
    "0.48": [704.0, 1472.0],
    "0.5": [704.0, 1408.0],
    "0.52": [704.0, 1344.0],
    "0.57": [768.0, 1344.0],
    "0.6": [768.0, 1280.0],
    "0.68": [836.3, 1229.3],
    "0.72": [836.3, 1152.0],
    "0.78": [897.3, 1152.0],
    "0.82": [897.3, 1088.0],
    "0.88": [960.0, 1088.0],
    "0.94": [960.0, 1024.0],
    "1.0": [1024.0, 1024.0],
    "1.07": [1024.0, 960.0],
    "1.13": [1088.0, 960.0],
    "1.21": [1088.0, 897.3],
    "1.29": [1152.0, 897.3],
    "1.38": [1152.0, 836.3],
    "1.46": [1218.7, 836.3],
    "1.67": [1280.0, 768.0],
    "1.75": [1344.0, 768.0],
    "2.0": [1408.0, 704.0],
    "2.09": [1472.0, 704.0],
    "2.4": [1536.0, 640.0],
    "2.5": [1600.0, 640.0],
    "2.89": [1664.0, 576.0],
    "3.0": [1728.0, 576.0],
    "3.11": [1797.3, 576.0],
    "3.62": [1852.3, 512.0],
    "3.75": [1920.0, 512.0],
    "3.88": [1988.8, 512.0],
    "4.0": [2048.0, 512.0],
}

ASPECT_RATIO_RANDOM_CROP_1024 = {
    "0.42": [640.0, 1536.0],
    "0.5": [704.0, 1408.0],
    "0.57": [768.0, 1344.0],
    "0.68": [836.3, 1229.3],
    "0.78": [897.3, 1152.0],
    "0.88": [960.0, 1088.0],
    "0.94": [960.0, 1024.0],
    "1.0": [1024.0, 1024.0],
    "1.07": [1024.0, 960.0],
    "1.13": [1088.0, 960.0],
    "1.29": [1152.0, 897.3],
    "1.46": [1218.7, 836.3],
    "1.75": [1344.0, 768.0],
    "2.0": [1408.0, 704.0],
    "2.4": [1536.0, 640.0],
}


ASPECT_RATIO_960 = {
    "0.25": [480.0, 1920.0],
    "0.26": [480.0, 1862.0],
    "0.27": [480.0, 1800.0],
    "0.28": [480.0, 1738.0],
    "0.32": [540.0, 1688.0],
    "0.33": [540.0, 1620.0],
    "0.35": [540.0, 1560.0],
    "0.4": [600.0, 1500.0],
    "0.42": [600.0, 1440.0],
    "0.48": [660.0, 1380.0],
    "0.5": [660.0, 1320.0],
    "0.52": [660.0, 1260.0],
    "0.57": [720.0, 1260.0],
    "0.6": [720.0, 1200.0],
    "0.68": [784.0, 1152.0],
    "0.72": [784.0, 1082.0],
    "0.78": [842.0, 1082.0],
    "0.82": [842.0, 1020.0],
    "0.88": [900.0, 1020.0],
    "0.94": [900.0, 960.0],
    "1.0": [960.0, 960.0],
    "1.07": [960.0, 900.0],
    "1.13": [1020.0, 900.0],
    "1.21": [1020.0, 842.0],
    "1.29": [1082.0, 842.0],
    "1.38": [1082.0, 784.0],
    "1.46": [1144.0, 784.0],
    "1.67": [1200.0, 720.0],
    "1.75": [1260.0, 720.0],
    "2.0": [1320.0, 660.0],
    "2.09": [1380.0, 660.0],
    "2.4": [1440.0, 600.0],
    "2.5": [1500.0, 600.0],
    "2.89": [1560.0, 540.0],
    "3.0": [1620.0, 540.0],
    "3.11": [1688.0, 540.0],
    "3.62": [1738.0, 480.0],
    "3.75": [1800.0, 480.0],
    "3.88": [1862.0, 480.0],
    "4.0": [1920.0, 480.0],
}

ASPECT_RATIO_RANDOM_CROP_960 = {
    "0.42": [600.0, 1440.0],
    "0.5": [660.0, 1320.0],
    "0.57": [720.0, 1260.0],
    "0.68": [784.0, 1152.0],
    "0.78": [842.0, 1082.0],
    "0.88": [900.0, 1020.0],
    "0.94": [900.0, 960.0],
    "1.0": [960.0, 960.0],
    "1.07": [960.0, 900.0],
    "1.13": [1020.0, 900.0],
    "1.29": [1082.0, 842.0],
    "1.46": [1144.0, 784.0],
    "1.75": [1260.0, 720.0],
    "2.0": [1320.0, 660.0],
    "2.4": [1440.0, 600.0],
}

ASPECT_RATIO_512 = {
    "0.25": [256.0, 1024.0],
    "0.26": [256.0, 992.0],
    "0.27": [256.0, 960.0],
    "0.28": [256.0, 928.0],
    "0.32": [288.0, 896.0],
    "0.33": [288.0, 864.0],
    "0.35": [288.0, 832.0],
    "0.4": [320.0, 800.0],
    "0.42": [320.0, 768.0],
    "0.48": [352.0, 736.0],
    "0.5": [352.0, 704.0],
    "0.52": [352.0, 672.0],
    "0.57": [384.0, 672.0],
    "0.6": [384.0, 640.0],
    "0.68": [416.0, 608.0],
    "0.72": [416.0, 576.0],
    "0.78": [448.0, 576.0],
    "0.82": [448.0, 544.0],
    "0.88": [480.0, 544.0],
    "0.94": [480.0, 512.0],
    "1.0": [512.0, 512.0],
    "1.07": [512.0, 480.0],
    "1.13": [544.0, 480.0],
    "1.21": [544.0, 448.0],
    "1.29": [576.0, 448.0],
    "1.38": [576.0, 416.0],
    "1.46": [608.0, 416.0],
    "1.67": [640.0, 384.0],
    "1.75": [672.0, 384.0],
    "2.0": [704.0, 352.0],
    "2.09": [736.0, 352.0],
    "2.4": [768.0, 320.0],
    "2.5": [800.0, 320.0],
    "2.89": [832.0, 288.0],
    "3.0": [864.0, 288.0],
    "3.11": [896.0, 288.0],
    "3.62": [928.0, 256.0],
    "3.75": [960.0, 256.0],
    "3.88": [992.0, 256.0],
    "4.0": [1024.0, 256.0],
}
ASPECT_RATIO_RANDOM_CROP_512 = {
    "0.42": [320.0, 768.0],
    "0.5": [352.0, 704.0],
    "0.57": [384.0, 672.0],
    "0.68": [416.0, 608.0],
    "0.78": [448.0, 576.0],
    "0.88": [480.0, 544.0],
    "0.94": [480.0, 512.0],
    "1.0": [512.0, 512.0],
    "1.07": [512.0, 480.0],
    "1.13": [544.0, 480.0],
    "1.29": [576.0, 448.0],
    "1.46": [608.0, 416.0],
    "1.75": [672.0, 384.0],
    "2.0": [704.0, 352.0],
    "2.4": [768.0, 320.0],
}
ASPECT_RATIO_RANDOM_CROP_PROB = [1, 2, 4, 4, 4, 4, 8, 8, 8, 4, 4, 4, 4, 2, 1]
ASPECT_RATIO_RANDOM_CROP_PROB = np.array(ASPECT_RATIO_RANDOM_CROP_PROB) / sum(
    ASPECT_RATIO_RANDOM_CROP_PROB
)


def get_closest_ratio(height: float, width: float, ratios: dict = ASPECT_RATIO_512):
    aspect_ratio = height / width
    closest_ratio = min(
        ratios.keys(), key=lambda ratio: abs(float(ratio) - aspect_ratio)
    )
    return ratios[closest_ratio], float(closest_ratio)


def get_image_size_without_loading(path):
    with Image.open(path) as img:
        return img.size  # (width, height)


class RandomSampler(Sampler[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.

    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """

    data_source: Sized
    replacement: bool

    def __init__(
        self,
        data_source: Sized,
        replacement: bool = False,
        num_samples: Optional[int] = None,
        generator=None,
    ) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator
        self._pos_start = 0

        if not isinstance(self.replacement, bool):
            raise TypeError(
                f"replacement should be a boolean value, but got replacement={self.replacement}"
            )

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                f"num_samples should be a positive integer value, but got num_samples={self.num_samples}"
            )

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(
                    high=n, size=(32,), dtype=torch.int64, generator=generator
                ).tolist()
            yield from torch.randint(
                high=n,
                size=(self.num_samples % 32,),
                dtype=torch.int64,
                generator=generator,
            ).tolist()
        else:
            for _ in range(self.num_samples // n):
                xx = torch.randperm(n, generator=generator).tolist()
                if self._pos_start >= n:
                    self._pos_start = 0
                print("xx top 10", xx[:10], self._pos_start)
                for idx in range(self._pos_start, n):
                    yield xx[idx]
                    self._pos_start = (self._pos_start + 1) % n
                self._pos_start = 0
            yield from torch.randperm(n, generator=generator).tolist()[
                : self.num_samples % n
            ]

    def __len__(self) -> int:
        return self.num_samples


class AspectRatioBatchImageSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        aspect_ratios (dict): The predefined aspect ratios.
    """

    def __init__(
        self,
        sampler: Sampler,
        dataset: Dataset,
        batch_size: int,
        train_folder: str = None,
        aspect_ratios: dict = ASPECT_RATIO_512,
        drop_last: bool = False,
        config=None,
        **kwargs,
    ) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError(
                "sampler should be an instance of ``Sampler``, " f"but got {sampler}"
            )
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(
                "batch_size should be a positive integer value, "
                f"but got batch_size={batch_size}"
            )
        self.sampler = sampler
        self.dataset = dataset
        self.train_folder = train_folder
        self.batch_size = batch_size
        self.aspect_ratios = aspect_ratios
        self.drop_last = drop_last
        self.config = config
        # buckets for each aspect ratio
        self._aspect_ratio_buckets = {ratio: [] for ratio in aspect_ratios}
        # [str(k) for k, v in aspect_ratios]
        self.current_available_bucket_keys = list(aspect_ratios.keys())

    def __iter__(self):
        for idx in self.sampler:
            try:
                image_dict = self.dataset[idx]

                width, height = image_dict.get("width", None), image_dict.get(
                    "height", None
                )
                if width is None or height is None:
                    image_id, name = image_dict["file_path"], image_dict["text"]
                    if self.train_folder is None:
                        image_dir = image_id
                    else:
                        image_dir = os.path.join(self.train_folder, image_id)

                    width, height = get_image_size_without_loading(image_dir)

                    ratio = height / width  # self.dataset[idx]
                else:
                    height = int(height)
                    width = int(width)
                    ratio = height / width  # self.dataset[idx]
            except Exception as e:
                print(e)
                continue
            # find the closest aspect ratio
            closest_ratio = min(
                self.aspect_ratios.keys(), key=lambda r: abs(float(r) - ratio)
            )
            if closest_ratio not in self.current_available_bucket_keys:
                continue
            bucket = self._aspect_ratio_buckets[closest_ratio]
            bucket.append(idx)
            # yield a batch of indices in the same aspect ratio group
            if len(bucket) == self.batch_size:
                yield bucket[:]
                del bucket[:]


class AspectRatioBatchSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        aspect_ratios (dict): The predefined aspect ratios.
    """

    def __init__(
        self,
        sampler: Sampler,
        dataset: Dataset,
        batch_size: int,
        video_folder: str = None,
        train_data_format: str = "webvid",
        aspect_ratios: dict = ASPECT_RATIO_512,
        drop_last: bool = False,
        config=None,
        **kwargs,
    ) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError(
                "sampler should be an instance of ``Sampler``, " f"but got {sampler}"
            )
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(
                "batch_size should be a positive integer value, "
                f"but got batch_size={batch_size}"
            )
        self.sampler = sampler
        self.dataset = dataset
        self.video_folder = video_folder
        self.train_data_format = train_data_format
        self.batch_size = batch_size
        self.aspect_ratios = aspect_ratios
        self.drop_last = drop_last
        self.config = config
        # buckets for each aspect ratio
        self._aspect_ratio_buckets = {ratio: [] for ratio in aspect_ratios}
        # [str(k) for k, v in aspect_ratios]
        self.current_available_bucket_keys = list(aspect_ratios.keys())

    def __iter__(self):
        for idx in self.sampler:
            try:
                video_dict = self.dataset[idx]
                width, more = video_dict.get("width", None), video_dict.get(
                    "height", None
                )

                if width is None or height is None:
                    if self.train_data_format == "normal":
                        video_id, name = video_dict["file_path"], video_dict["text"]
                        if self.video_folder is None:
                            video_dir = video_id
                        else:
                            video_dir = os.path.join(self.video_folder, video_id)
                    else:
                        videoid, name, page_dir = (
                            video_dict["videoid"],
                            video_dict["name"],
                            video_dict["page_dir"],
                        )
                        video_dir = os.path.join(self.video_folder, f"{videoid}.mp4")
                    cap = cv2.VideoCapture(video_dir)

                    # 获取视频尺寸
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 浮点数转换为整数
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 浮点数转换为整数

                    ratio = height / width  # self.dataset[idx]
                else:
                    height = int(height)
                    width = int(width)
                    ratio = height / width  # self.dataset[idx]
            except Exception as e:
                print(e)
                continue
            # find the closest aspect ratio
            closest_ratio = min(
                self.aspect_ratios.keys(), key=lambda r: abs(float(r) - ratio)
            )
            if closest_ratio not in self.current_available_bucket_keys:
                continue
            bucket = self._aspect_ratio_buckets[closest_ratio]
            bucket.append(idx)
            # yield a batch of indices in the same aspect ratio group
            if len(bucket) == self.batch_size:
                yield bucket[:]
                del bucket[:]


class AspectRatioBatchImageVideoSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        aspect_ratios (dict): The predefined aspect ratios.
    """

    def __init__(
        self,
        sampler: Sampler,
        dataset: Dataset,
        batch_size: int,
        train_folder: str = None,
        aspect_ratios: dict = ASPECT_RATIO_512,
        drop_last: bool = False,
    ) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError(
                "sampler should be an instance of ``Sampler``, " f"but got {sampler}"
            )
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(
                "batch_size should be a positive integer value, "
                f"but got batch_size={batch_size}"
            )
        self.sampler = sampler
        self.dataset = dataset
        self.train_folder = train_folder
        self.batch_size = batch_size
        self.aspect_ratios = aspect_ratios
        self.drop_last = drop_last

        # buckets for each aspect ratio
        self.current_available_bucket_keys = list(aspect_ratios.keys())
        self.bucket = {
            "image": {ratio: [] for ratio in aspect_ratios},
            "video": {ratio: [] for ratio in aspect_ratios},
        }

    def __iter__(self):
        for idx in self.sampler:
            content_type = self.dataset[idx].get("type", "image")
            if content_type == "image":
                try:
                    image_dict = self.dataset[idx]

                    width, height = image_dict.get("width", None), image_dict.get(
                        "height", None
                    )
                    if width is None or height is None:
                        image_id, name = image_dict["file_path"], image_dict["text"]
                        if self.train_folder is None:
                            image_dir = image_id
                        else:
                            image_dir = os.path.join(self.train_folder, image_id)

                        width, height = get_image_size_without_loading(image_dir)

                        ratio = height / width  # self.dataset[idx]
                    else:
                        height = int(height)
                        width = int(width)
                        ratio = height / width  # self.dataset[idx]
                except Exception as e:
                    print(e)
                    continue
                # find the closest aspect ratio
                closest_ratio = min(
                    self.aspect_ratios.keys(), key=lambda r: abs(float(r) - ratio)
                )
                if closest_ratio not in self.current_available_bucket_keys:
                    continue
                bucket = self.bucket["image"][closest_ratio]
                bucket.append(idx)
                # yield a batch of indices in the same aspect ratio group
                if len(bucket) == self.batch_size:
                    yield bucket[:]
                    del bucket[:]
            else:
                try:
                    video_dict = self.dataset[idx]
                    width, height = video_dict.get("width", None), video_dict.get(
                        "height", None
                    )

                    if width is None or height is None:
                        video_id, name = video_dict["file_path"], video_dict["text"]
                        if self.train_folder is None:
                            video_dir = video_id
                        else:
                            video_dir = os.path.join(self.train_folder, video_id)
                        cap = cv2.VideoCapture(video_dir)

                        # 获取视频尺寸
                        width = int(
                            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                        )  # 浮点数转换为整数
                        height = int(
                            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                        )  # 浮点数转换为整数

                        ratio = height / width  # self.dataset[idx]
                    else:
                        height = int(height)
                        width = int(width)
                        ratio = height / width  # self.dataset[idx]
                except Exception as e:
                    print(e)
                    continue
                # find the closest aspect ratio
                closest_ratio = min(
                    self.aspect_ratios.keys(), key=lambda r: abs(float(r) - ratio)
                )
                if closest_ratio not in self.current_available_bucket_keys:
                    continue
                bucket = self.bucket["video"][closest_ratio]
                bucket.append(idx)
                # yield a batch of indices in the same aspect ratio group
                if len(bucket) == self.batch_size:
                    yield bucket[:]
                    del bucket[:]
