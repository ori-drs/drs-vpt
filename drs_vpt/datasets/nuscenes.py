# Adapted from MapAnything (https://github.com/facebookresearch/map-anything), mapanything/datasets/wai/ (dataset classes).
# Copyright (c) Meta Platforms, Inc. and affiliates. Apache License 2.0, see drs_vpt/_vendor/mapanything/LICENSE.
# Modified for DRS VPT (Oxford Robotics Institute): builds the point cloud map for each sample (LiDAR scans or depth-derived clouds, with scan and voxel-size augmentation) and its Sonata input.

"""
nuScenes dataset in WAI format, with the point cloud map DRS VPT localizes in.
"""

import os

import numpy as np

from drs_vpt.datasets.cloud import make_cloud_input
from drs_vpt._vendor.mapanything.datasets.base.base_dataset import BaseDataset
from drs_vpt.datasets.utils import ensure_orientation_from_prescribed_resolution_ccw, \
    apply_transform_from_left
from drs_vpt._vendor.mapanything.utils.wai.core import load_data, load_frame


class NuscenesWAI(BaseDataset):
    """
    OSD dataset containing outdoor phototourism and in-the-wild scenes.
    Also includes Tanks & Temples scenes.
    """

    def __init__(
        self,
        *args,
        ROOT,
        dataset_metadata_dir,
        split,
        overfit_num_sets=None,
        sample_specific_scene: bool = False,
        specific_scene_name: str = None,
        **kwargs,
    ):
        """
        Initialize the dataset attributes.
        Args:
            ROOT: Root directory of the dataset.
            dataset_metadata_dir: Path to the dataset metadata directory.
            split: Dataset split (train, val, test).
            overfit_num_sets: If None, use all sets. Else, the dataset will be truncated to this number of sets.
            sample_specific_scene: Whether to sample a specific scene from the dataset.
            specific_scene_name: Name of the specific scene to sample.
        """
        # Initialize the dataset attributes
        super().__init__(*args, **kwargs)
        self.ROOT = ROOT
        self.dataset_metadata_dir = dataset_metadata_dir
        self.split = split
        self.overfit_num_sets = overfit_num_sets
        self.sample_specific_scene = sample_specific_scene
        self.specific_scene_name = specific_scene_name
        self._load_data()

        # Define the dataset type flags
        self.is_metric_scale = True
        self.is_synthetic = False

    def _load_data(self):
        "Load the precomputed dataset metadata"
        # Load the dataset metadata corresponding to the split
        split_metadata_path = os.path.join(
            self.dataset_metadata_dir,
            self.split,
            f"nuscenes_scene_list_{self.split}.npy",
        )
        split_scene_list = np.load(split_metadata_path, allow_pickle=True)

        # Get the list of all scenes
        if not self.sample_specific_scene:
            self.scenes = list(split_scene_list)
        else:
            self.scenes = [self.specific_scene_name]
        self.num_of_scenes = len(self.scenes)

    def _get_views(self, sampled_idx, num_views_to_sample, resolution):
        cloud_cfg = self.cloud_config
        # Get the scene name of the sampled index
        scene_index = sampled_idx
        scene_name = self.scenes[scene_index]

        # Get the metadata corresponding to the scene
        scene_root = os.path.join(self.ROOT, scene_name)
        scene_meta = load_data(
            os.path.join(scene_root, "scene_meta.json"), "scene_meta"
        )

        scene_file_names = list(scene_meta["frame_names"].keys())

        # Load the scene pairwise covisibility mmap
        covisibility_version_key = "v0"
        covisibility_map_dir = os.path.join(
            scene_root, "covisibility", covisibility_version_key
        )
        # Assumes only npy file in directory is covisbility map
        covisibility_map_name = next(
            f for f in os.listdir(covisibility_map_dir) if f.endswith(".npy")
        )
        covisibility_map_path = os.path.join(
            scene_root, "covisibility", covisibility_version_key, covisibility_map_name
        )
        pairwise_covisibility = load_data(covisibility_map_path, "mmap")

        # Get the indices of the N views in the scene
        image_indices, cloud_indices = self._sample_image_and_cloud_frames(
            num_views_to_sample, scene_file_names, pairwise_covisibility
        )

        # Get the views corresponding to the selected view indices
        views = []
        for view_index in image_indices:
            # Load the data corresponding to the view
            view_file_name = scene_file_names[view_index]
            view_data = load_frame(
                scene_root,
                view_file_name,
                modalities=["image", "depth"],
                scene_meta=scene_meta,
            )

            # Convert necessary data to numpy
            image = view_data["image"].permute(1, 2, 0).numpy()
            image = (image * 255).astype(np.uint8)
            depthmap = view_data["depth"].numpy().astype(np.float32)
            intrinsics = view_data["intrinsics"].numpy().astype(np.float32)
            c2w_pose = view_data["extrinsics"].numpy().astype(np.float32)

            image, depthmap, intrinsics, c2w_pose, rotated = ensure_orientation_from_prescribed_resolution_ccw(
                image, depthmap, intrinsics, c2w_pose,
                prescribed_resolution_wh=resolution
            )

            # Ensure that the depthmap has all valid values
            depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0)

            # Resize the data to match the desired resolution
            image, depthmap, intrinsics = (
                self._crop_resize_if_necessary(
                    image=image,
                    resolution=resolution,
                    depthmap=depthmap,
                    intrinsics=intrinsics
                )
            )

            view = dict(
                img=image,
                depthmap=depthmap,
                camera_pose=c2w_pose,  # cam2world
                camera_intrinsics=intrinsics,
                dataset="Nuscenes",
                label=scene_name,
                instance=os.path.join("images", str(view_file_name)))

            # Append the view dictionary to the list of views
            views.append(view)

        clouds_in_world_frame = list()
        cloud_camera_pose = list()
        for view_index in cloud_indices:
            # Load the data corresponding to the view
            view_file_name = scene_file_names[view_index]
            view_data = load_frame(
                scene_root,
                view_file_name,
                modalities=["cloud"],
                scene_meta=scene_meta,
            )
            cloud_in_camera_frame = view_data["cloud"].astype(np.float32)
            ranges = np.linalg.norm(cloud_in_camera_frame, axis=-1)
            cloud_in_camera_frame = cloud_in_camera_frame[ranges<80]
            c2w_pose = view_data["extrinsics"].numpy().astype(np.float32)

            cloud_in_world_frame = apply_transform_from_left(c2w_pose, cloud_in_camera_frame)
            clouds_in_world_frame.append(cloud_in_world_frame)
            cloud_camera_pose.append(c2w_pose)

        random_resolution = np.random.uniform(*cloud_cfg.grid_size)
        grid_size = np.array([random_resolution]).astype(np.float32)
        full_cloud_in_world_frame = np.vstack(clouds_in_world_frame)
        cloud = make_cloud_input(full_cloud_in_world_frame, cloud_camera_pose[0], grid_size, cloud_cfg.scanner_offset,
                                 cloud_cfg.max_points)
        return views, cloud
