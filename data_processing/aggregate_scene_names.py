# Adapted from MapAnything (https://github.com/facebookresearch/map-anything), data_processing/aggregate_scene_names.py.
# Copyright (c) Meta Platforms, Inc. and affiliates. Apache License 2.0, see drs_vpt/_vendor/mapanything/LICENSE.
# Modified for DRS VPT (Oxford Robotics Institute): the DRS VPT training datasets only.

"""
Script to aggregate the scene names for the different splits used in MapAnything.
The valid scenes for each dataset are organized into train, val and optionally test splits.
Scene lists are saved as numpy arrays for efficient loading.
"""

import argparse
import os
from typing import List, Optional, Set, Union

import numpy as np
from natsort import natsorted
from tqdm import tqdm

from drs_vpt._vendor.mapanything.datasets.utils.data_splits import (
    MegaDepthSplits,
    ETH3DSplits,
    OSDSplits,
    KITTISplits,
    NuscenesSplits,
    SevenScenesSplits,
    ArgoverseSplits,
    WildRGBDSplits,
    ScannetPPVGGSplits,
    GrandTourSplits,
    PandasetSplits,
)
from drs_vpt._vendor.mapanything.utils.misc import seed_everything
from drs_vpt._vendor.mapanything.utils.wai.scene_frame import get_scene_names


def save_scene_lists(scene_list, output_path):
    """Save the list of scene names as a numpy array."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert scene list to numpy array and save
    scene_array = np.array(scene_list, dtype=object)
    np.save(output_path, scene_array)

    print(f"Saved {len(scene_list)} scene names to {output_path}")


def print_dataset_stats(output_dir: str, datasets: List[str]):
    """Print statistics for the number of scenes in each dataset split.

    Args:
        output_dir: Path to the output directory containing the aggregated scene lists
        datasets: List of dataset names to print stats for
    """
    print("\n" + "=" * 80)
    print("DATASET SCENE STATISTICS")
    print("=" * 80)

    # Define the possible splits
    splits = ["train", "val", "test"]

    # Track totals across all datasets
    total_stats = {split: 0 for split in splits}

    for dataset in datasets:
        print(f"\n{dataset.upper()}:")
        print("-" * 40)

        dataset_stats = {}
        dataset_total = 0

        for split in splits:
            split_file = os.path.join(
                output_dir, split, f"{dataset}_scene_list_{split}.npy"
            )

            if os.path.exists(split_file):
                try:
                    scene_array = np.load(split_file, allow_pickle=True)
                    num_scenes = len(scene_array)
                    dataset_stats[split] = num_scenes
                    dataset_total += num_scenes
                    total_stats[split] += num_scenes
                    print(f"  {split:>5}: {num_scenes:>6} scenes")
                except Exception as e:
                    print(f"  {split:>5}: Error loading file - {e}")
                    dataset_stats[split] = 0
            else:
                dataset_stats[split] = 0

        if dataset_total > 0:
            print(f"  {'Total':>5}: {dataset_total:>6} scenes")
        else:
            print(f"  No scene files found for {dataset}")

    # Print overall statistics
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)

    overall_total = 0
    for split in splits:
        if total_stats[split] > 0:
            print(f"  {split:>5}: {total_stats[split]:>6} scenes")
            overall_total += total_stats[split]

    print(f"  {'Total':>5}: {overall_total:>6} scenes")
    print("=" * 80)


class DatasetAggregator:
    """Base class for dataset scene list aggregation."""

    def __init__(
        self,
        dataset_name: str,
        root_dir: str,
        output_dir: str,
        depth_folder: str = "depth",
        covisibility_version_key: str = "v0",
        raw_data_root_dir: Optional[str] = None,
    ):
        """Initialize the dataset aggregator.

        Args:
            dataset_name: Name of the dataset
            root_dir: Path to the root directory of the dataset
            output_dir: Path to the output directory
            depth_folder: Name of the depth folder (usually "depth", "rendered_depth" or "pred_depth/method")
            covisibility_version_key: Key for the covisibility version
            raw_data_root_dir: Path to the raw data root directory (optional)
        """
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.depth_folder = depth_folder
        self.covisibility_version_key = covisibility_version_key
        self.raw_data_root_dir = raw_data_root_dir

        # Get all WAI scenes
        self.all_scenes = self._get_all_scenes()

    def _get_all_scenes(self) -> Set[str]:
        """Get all WAI scenes for the dataset."""
        cfg = {
            "root": self.root_dir,
            "scene_filters": [
                {"exists": "scene_meta.json"},
                {"exists": f"covisibility/{self.covisibility_version_key}"},
                {"exists": self.depth_folder},
            ],
        }
        return set(get_scene_names(cfg))

    def _get_split_scenes(
        self, split_name: str, split_scenes: Union[List[str], str, Set[str]]
    ) -> List[str]:
        """Get the scenes for a specific split."""
        if split_scenes == "all":
            return_scenes = natsorted(list(self.all_scenes))
            self.all_scenes -= self.all_scenes
            return return_scenes

        if isinstance(split_scenes, (list, set)):
            split_set = set(split_scenes)
            if not split_set.issubset(self.all_scenes):
                missing_scenes = split_set - self.all_scenes
                print(
                    f"{len(missing_scenes)} {split_name} scenes are missing in all WAI scenes of {self.dataset_name}."
                )
                split_set -= missing_scenes
            self.all_scenes -= split_set
            return natsorted(list(split_set))

        raise ValueError(f"Invalid split_scenes type: {type(split_scenes)}")

    def _process_split(self, split_name: str, scenes: List[str]):
        """Process a specific split."""
        print(f"Saving {split_name} split scene list...")
        split_output_dir = os.path.join(self.output_dir, split_name)
        os.makedirs(split_output_dir, exist_ok=True)
        output_path = os.path.join(
            split_output_dir, f"{self.dataset_name}_scene_list_{split_name}.npy"
        )
        save_scene_lists(scenes, output_path)

    def aggregate(
        self, val_split_scenes=None, test_split_scenes=None, train_split_scenes=None
    ):
        """Aggregate the dataset scene lists."""
        # Process test split if provided
        if test_split_scenes is not None:
            test_scenes = self._get_split_scenes("test", test_split_scenes)
            self._process_split("test", test_scenes)

        # Process val split if provided
        if val_split_scenes is not None:
            val_scenes = self._get_split_scenes("val", val_split_scenes)
            self._process_split("val", val_scenes)

        # Process remaining scenes as train split if not provided
        if train_split_scenes is not None:
            train_scenes = self._get_split_scenes("train", train_split_scenes)
            self._process_split("train", train_scenes)
        else:
            if self.all_scenes and (len(self.all_scenes) > 0):
                train_scenes = natsorted(list(self.all_scenes))
                self._process_split("train", train_scenes)


class ETH3DAggregator(DatasetAggregator):
    """Aggregator for ETH3D dataset."""

    def __init__(self, root_dir, output_dir, covisibility_version_key="v0"):
        super().__init__(
            dataset_name="eth3d",
            root_dir=root_dir,
            output_dir=output_dir,
            covisibility_version_key=covisibility_version_key,
            depth_folder="depth",
        )
        self.dataset_split_info = ETH3DSplits()

    def aggregate(self):
        """Aggregate the ETH3D dataset."""
        super().aggregate(test_split_scenes=self.dataset_split_info.test_split_scenes)


class MegaDepthAggregator(DatasetAggregator):
    """Aggregator for MegaDepth dataset."""

    def __init__(self, root_dir, output_dir, covisibility_version_key="v0"):
        super().__init__(
            dataset_name="megadepth",
            root_dir=root_dir,
            output_dir=output_dir,
            covisibility_version_key=covisibility_version_key,
            depth_folder="depth",
        )
        self.dataset_split_info = MegaDepthSplits()

    def aggregate(self):
        """Aggregate the MegaDepth dataset."""
        super().aggregate(val_split_scenes=self.dataset_split_info.val_split_scenes)


class OSDAggregator(DatasetAggregator):
    """Aggregator for MegaDepth dataset."""

    def __init__(self, root_dir, output_dir, covisibility_version_key="v0"):
        super().__init__(
            dataset_name="osd",
            root_dir=root_dir,
            output_dir=output_dir,
            covisibility_version_key=covisibility_version_key,
            depth_folder="depth",
        )
        self.dataset_split_info = OSDSplits()

    def aggregate(self):
        """Aggregate the MegaDepth dataset."""
        super().aggregate(val_split_scenes=self.dataset_split_info.val_split_scenes)


class KITTIAggregator(DatasetAggregator):
    """Aggregator for MegaDepth dataset."""

    def __init__(self, root_dir, output_dir, covisibility_version_key="v0"):
        super().__init__(
            dataset_name="kitti",
            root_dir=root_dir,
            output_dir=output_dir,
            covisibility_version_key=covisibility_version_key,
            depth_folder="depth",
        )
        self.dataset_split_info = KITTISplits()

    def aggregate(self):
        """Aggregate the MegaDepth dataset."""
        super().aggregate(val_split_scenes=self.dataset_split_info.val_split_scenes)


class NuscenesAggregator(DatasetAggregator):
    """Aggregator for MegaDepth dataset."""

    def __init__(self, root_dir, output_dir, covisibility_version_key="v0"):
        super().__init__(
            dataset_name="nuscenes",
            root_dir=root_dir,
            output_dir=output_dir,
            covisibility_version_key=covisibility_version_key,
            depth_folder="depth",
        )
        self.dataset_split_info = NuscenesSplits()

    def aggregate(self):
        """Aggregate the MegaDepth dataset."""
        super().aggregate(val_split_scenes=self.dataset_split_info.val_split_scenes)


class SevenScenesAggregator(DatasetAggregator):
    """Aggregator for MegaDepth dataset."""

    def __init__(self, root_dir, output_dir, covisibility_version_key="v0"):
        super().__init__(
            dataset_name="7scenes",
            root_dir=root_dir,
            output_dir=output_dir,
            covisibility_version_key=covisibility_version_key,
            depth_folder="depth",
        )
        self.dataset_split_info = SevenScenesSplits()

    def aggregate(self):
        """Aggregate the MegaDepth dataset."""
        super().aggregate(val_split_scenes=self.dataset_split_info.val_split_scenes)


class ArgoverseAggregator(DatasetAggregator):
    """Aggregator for MegaDepth dataset."""

    def __init__(self, root_dir, output_dir, covisibility_version_key="v0"):
        super().__init__(
            dataset_name="argoverse",
            root_dir=root_dir,
            output_dir=output_dir,
            covisibility_version_key=covisibility_version_key,
            depth_folder="depth",
        )
        self.dataset_split_info = ArgoverseSplits()

    def aggregate(self):
        """Aggregate the MegaDepth dataset."""
        super().aggregate(val_split_scenes=self.dataset_split_info.val_split_scenes)


class WildRGBDAggregator(DatasetAggregator):
    """Aggregator for MegaDepth dataset."""

    def __init__(self, root_dir, output_dir, covisibility_version_key="v0"):
        super().__init__(
            dataset_name="wildrgbd",
            root_dir=root_dir,
            output_dir=output_dir,
            covisibility_version_key=covisibility_version_key,
            depth_folder="depth",
        )
        self.dataset_split_info = WildRGBDSplits()

    def aggregate(self):
        """Aggregate the MegaDepth dataset."""
        super().aggregate(val_split_scenes=self.dataset_split_info.val_split_scenes)


class ScannetPPVGGAggregator(DatasetAggregator):
    """Aggregator for MegaDepth dataset."""

    def __init__(self, root_dir, output_dir, covisibility_version_key="v0"):
        super().__init__(
            dataset_name="scannetppvgg",
            root_dir=root_dir,
            output_dir=output_dir,
            covisibility_version_key=covisibility_version_key,
            depth_folder="depth",
        )
        self.dataset_split_info = ScannetPPVGGSplits()

    def aggregate(self):
        """Aggregate the MegaDepth dataset."""
        super().aggregate(val_split_scenes=self.dataset_split_info.val_split_scenes)


class GrandTourAggregator(DatasetAggregator):
    """Aggregator for MegaDepth dataset."""

    def __init__(self, root_dir, output_dir, covisibility_version_key="v0"):
        super().__init__(
            dataset_name="grand_tour",
            root_dir=root_dir,
            output_dir=output_dir,
            covisibility_version_key=covisibility_version_key,
            depth_folder="depth",
        )
        self.dataset_split_info = GrandTourSplits()

    def aggregate(self):
        """Aggregate the MegaDepth dataset."""
        super().aggregate(val_split_scenes=self.dataset_split_info.val_split_scenes)


class PandasetAggregator(DatasetAggregator):
    """Aggregator for MegaDepth dataset."""

    def __init__(self, root_dir, output_dir, covisibility_version_key="v0"):
        super().__init__(
            dataset_name="pandaset",
            root_dir=root_dir,
            output_dir=output_dir,
            covisibility_version_key=covisibility_version_key,
            depth_folder="depth",
        )
        self.dataset_split_info = PandasetSplits()

    def aggregate(self):
        """Aggregate the MegaDepth dataset."""
        super().aggregate(val_split_scenes=self.dataset_split_info.val_split_scenes)


AGGREGATORS = {
    "megadepth": MegaDepthAggregator,
    "eth3d": ETH3DAggregator,
    "osd": OSDAggregator,
    "kitti": KITTIAggregator,
    "nuscenes": NuscenesAggregator,
    "7scenes": SevenScenesAggregator,
    "argoverse": ArgoverseAggregator,
    "wildrgbd": WildRGBDAggregator,
    "scannetppvgg": ScannetPPVGGAggregator,
    "grand_tour": GrandTourAggregator,
    "pandaset": PandasetAggregator,
}


def main():
    """Write <output_dir>/<split>/<dataset>_scene_list_<split>.npy for each dataset."""
    seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--wai_root", type=str, required=True, help="Root of the WAI-format datasets")
    parser.add_argument("--output_dir", type=str, required=True, help="Metadata directory (machine.metadata_dir)")
    parser.add_argument("--datasets", type=str, nargs="+", default=None, choices=list(AGGREGATORS),
                        help="Datasets to process (default: every dataset with a folder under --wai_root)")
    parser.add_argument("--print_stats", action="store_true",
                        help="Print the number of scenes in each dataset split and exit")
    args = parser.parse_args()

    if args.datasets is None:
        args.datasets = [d for d in AGGREGATORS if os.path.isdir(os.path.join(args.wai_root, d))]
        print(f"Datasets found under {args.wai_root}: {', '.join(args.datasets) or 'none'}")

    os.makedirs(args.output_dir, exist_ok=True)
    if args.print_stats:
        print_dataset_stats(args.output_dir, args.datasets)
        return

    for dataset in tqdm(args.datasets):
        AGGREGATORS[dataset](root_dir=os.path.join(args.wai_root, dataset), output_dir=args.output_dir).aggregate()


if __name__ == "__main__":
    main()
