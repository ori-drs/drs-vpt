# Adapted from MapAnything (https://github.com/facebookresearch/map-anything), mapanything/datasets/__init__.py.
# Copyright (c) Meta Platforms, Inc. and affiliates. Apache License 2.0, see drs_vpt/_vendor/mapanything/LICENSE.
# Modified for DRS VPT (Oxford Robotics Institute): collates each view's point cloud into a Sonata batch
# (collate_points_and_images), and dataset strings are evaluated against the DRS VPT training datasets.

"""
Data loaders for DRS VPT training. A dataset is given as a string, e.g. "10_000 @ KITTIWAI(split='train', ...)",
and evaluated against the classes imported here.
"""

import torch
from torch.utils.data import default_collate

from drs_vpt._vendor import sonata
from drs_vpt._vendor.mapanything.datasets.wai.eth3d import ETH3DWAI  # noqa
from drs_vpt._vendor.mapanything.datasets.wai.megadepth import MegaDepthWAI  # noqa
from drs_vpt.datasets.argoverse import ArgoverseWAI  # noqa
from drs_vpt.datasets.grand_tour import GrandTourWAI  # noqa
from drs_vpt.datasets.kitti import KITTIWAI  # noqa
from drs_vpt.datasets.nuscenes import NuscenesWAI  # noqa
from drs_vpt.datasets.osd import OSDWAI  # noqa
from drs_vpt.datasets.pandaset import PandasetWAI  # noqa
from drs_vpt.datasets.scannetppvgg import ScannetPPVGGWAI  # noqa
from drs_vpt.datasets.seven_scenes import SevenScenesWAI  # noqa
from drs_vpt.datasets.wildrgbd import WildRGBDWAI  # noqa
from drs_vpt.train.train_tools import get_rank, get_world_size

CLOUD_KEYS = {"coord", "normal", "color", "grid_size", "scene_scale"}  # the Sonata input fields of a sample's cloud


def get_test_data_loader(
    dataset, batch_size, num_workers=8, shuffle=False, drop_last=False, pin_mem=True
):
    "Get simple PyTorch dataloader corresponding to the testing dataset"
    # PyTorch dataset
    if isinstance(dataset, str):
        dataset = eval(dataset)

    world_size = get_world_size()
    rank = get_rank()

    if torch.distributed.is_initialized():
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )
    elif shuffle:
        sampler = torch.utils.data.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=drop_last,
        collate_fn=collate_points_and_images,
    )

    return data_loader


class DynamicBatchDatasetWrapper:
    """
    Wrapper dataset that handles DynamicBatchedMultiFeatureRandomSampler output.

    The dynamic sampler returns batches (lists of tuples) instead of individual samples.
    This wrapper ensures that the underlying dataset's __getitem__ method gets called
    with individual tuples as expected.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, batch_indices):
        """
        Handle batch of indices from DynamicBatchedMultiFeatureRandomSampler.

        Args:
            batch_indices: List of tuples like [(sample_idx, feat_idx_1, feat_idx_2, ...), ...]

        Returns:
            List of samples from the underlying dataset
        """
        if isinstance(batch_indices, (list, tuple)) and len(batch_indices) > 0:
            # If it's a batch (list of tuples), process each item
            if isinstance(batch_indices[0], (list, tuple)):
                return [self.dataset[idx] for idx in batch_indices]
            else:
                # Single tuple, call dataset directly
                return self.dataset[batch_indices]
        else:
            # Fallback for single index
            return self.dataset[batch_indices]

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        # Delegate all other attributes to the wrapped dataset
        return getattr(self.dataset, name)


def get_train_data_loader(
    dataset,
    max_num_of_imgs_per_gpu,
    num_workers=8,
    shuffle=True,
    drop_last=True,
    pin_mem=True,
):
    "Dynamic PyTorch dataloader corresponding to the training dataset"
    # PyTorch dataset
    if isinstance(dataset, str):
        dataset = eval(dataset)

    world_size = get_world_size()
    rank = get_rank()

    # Get DynamicBatchedMultiFeatureRandomSampler
    batch_sampler = dataset.make_sampler(
        shuffle=shuffle,
        world_size=world_size,
        rank=rank,
        drop_last=drop_last,
        max_num_of_images_per_gpu=max_num_of_imgs_per_gpu,
        use_dynamic_sampler=True,
    )

    # Wrap the dataset to handle batch format from dynamic sampler
    wrapped_dataset = DynamicBatchDatasetWrapper(dataset)

    # Init the dynamic data loader
    data_loader = torch.utils.data.DataLoader(
        wrapped_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_mem,
        collate_fn=collate_points_and_images,
    )

    return data_loader


def collate_points_and_images(samples):
    """Batch {"image_views": [...], "cloud": {...}} samples: the image views with default_collate (one dict per view
    position), the clouds with Sonata's transform and collate, plus their scanner poses."""
    image_views = default_collate([sample["image_views"] for sample in samples])
    clouds = [sample["cloud"] for sample in samples]
    point_transform = sonata.transform.default()
    cloud = sonata.data.collate_fn([point_transform({k: c[k] for k in CLOUD_KEYS}) for c in clouds])
    cloud["scanner_pose"] = default_collate([c["scanner_pose"] for c in clouds])
    return {"image_views": image_views, "cloud": cloud}
