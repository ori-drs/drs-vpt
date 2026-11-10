# Adapted from MapAnything (https://github.com/facebookresearch/map-anything), mapanything/datasets/utils/data_splits.py.
# Copyright (c) Meta Platforms, Inc. and affiliates. Apache License 2.0, see drs_vpt/_vendor/mapanything/LICENSE.
# Modified for DRS VPT (Oxford Robotics Institute): only the DRS VPT training datasets; splits for the point cloud datasets added.

"""
Modules containing dataset split information
"""


class MegaDepthSplits:
    """
    This class contains the information about the splits of the MegaDepth dataset.
    """

    def __init__(self):
        """
        Validation split is based on scenes used in DUSt3R.
        """
        self.val_split_scenes = ["0015_0", "0015_1", "0022_0"]


class ETH3DSplits:
    """
    This class contains the information about the splits of the ETH3D dataset.
    """

    def __init__(self):
        """
        All scenes are in the test split.
        """
        self.test_split_scenes = "all"


class OSDSplits:
    """
    This class contains the information about the splits of the MegaDepth dataset.
    """

    def __init__(self):
        """
        Validation split is based on scenes used in DUSt3R.
        """
        self.val_split_scenes = ["2024-03-12-keble-college-01_0004",
                                 "2024-03-12-keble-college-02_0002",
                                 "2024-03-18-christ-church-04_0002",
                                 "2024-03-15-bodleian-library-01_0007",
                                 "2024-03-15-bodleian-library-01_0001",
                                 "2024-03-18-christ-church-01_0005",
                                 "2024-03-18-christ-church-01_0001",
                                 ]


class KITTISplits:
    """
    This class contains the information about the splits of the MegaDepth dataset.
    """

    def __init__(self):
        """
        Validation split is based on scenes used in DUSt3R.
        """
        scene_name = "00"
        splits = [f"{scene_name}_{x:04d}" for x in range(46)]
        self.val_split_scenes = splits


class NuscenesSplits:

    def __init__(self):
        splits = [f"train_{x:04d}" for x in range(100)[::4]]
        self.val_split_scenes = splits


class SevenScenesSplits:

    def __init__(self):
        splits = [f"train_{x:04d}" for x in range(100)[::4]]

        seq1_scenes = ["chess", "fire", "office", "redkitchen" ]
        all_scenes = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
        seq2_scenes = list(set(all_scenes).difference(seq1_scenes))

        splits = [f"{x}_seq-01_0000" for x in seq1_scenes] + [f"{x}_seq-02_0000" for x in seq2_scenes]

        self.val_split_scenes = splits


class ArgoverseSplits:
    """
    This class contains the information about the splits of the MegaDepth dataset.
    """

    def __init__(self):
        """
        Validation split is based on scenes used in DUSt3R.
        """
        self.val_split_scenes = [
            "0ef28d5c-ae34-370b-99e7-6709e1c4b929_0000",
            "0ef28d5c-ae34-370b-99e7-6709e1c4b929_0001",
            "0ef28d5c-ae34-370b-99e7-6709e1c4b929_0002",
            "5c251c22-11b2-3278-835c-0cf3cdee3f44_0000",
            "5c251c22-11b2-3278-835c-0cf3cdee3f44_0001",
            "5c251c22-11b2-3278-835c-0cf3cdee3f44_0002",
                                 ]


class WildRGBDSplits:
    """
    This class contains the information about the splits of the MegaDepth dataset.
    """

    def __init__(self):
        """
        Validation split is based on scenes used in DUSt3R.
        """
        self.val_split_scenes = ["apple_scene_002"]


class ScannetPPVGGSplits:
    """
    This class contains the information about the splits of the MegaDepth dataset.
    """

    def __init__(self):
        """
        Validation split is based on scenes used in DUSt3R.
        """
        self.val_split_scenes = ["0a5c013435"]


class GrandTourSplits:
    """
    This class contains the information about the splits of the MegaDepth dataset.
    """

    def __init__(self):
        """
        Validation split is based on scenes used in DUSt3R.
        """
        self.val_split_scenes = ["2024-11-04-11-56-53_0002", "2024-11-14-12-01-26_0002", "2024-12-09-11-53-11_0003"]


class PandasetSplits:
    """
    This class contains the information about the splits of the MegaDepth dataset.
    """

    def __init__(self):
        """
        Validation split is based on scenes used in DUSt3R.
        """
        self.val_split_scenes = ["042_0000", "042_0001",
                                 "047_0000", "047_0001"]
