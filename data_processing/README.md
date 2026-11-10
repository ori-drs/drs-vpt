# Training data

DRS VPT trains on datasets converted to the WAI format (the format MapAnything uses). For each dataset: download it,
convert it, compute covisibility, then build the scene lists once for all of them. Every stage is restartable: a scene
that has already finished a stage is skipped.

Training reads two paths, set in `configs/machine/local.yaml`: `root_data_dir` (the WAI root, one folder per dataset,
`/data/wai` below) and `metadata_dir` (the scene lists, `/data/wai_meta` below).

None of the datasets can be redistributed here; download each one from its own site under its own licence.

## Install

Use a separate environment for processing. `wai_processing` and `drs_vpt` must both be importable.

```bash
sudo apt-get install -y libegl1 libgomp1 libusb-1.0-0     # open3d's system libraries, if missing (headless machines)
conda create -n drs_vpt_data python=3.12 -y && conda activate drs_vpt_data
pip install torch torchvision
pip install -e "data_processing/wai_processing[moge]"   # [moge] is only needed for MegaDepth
export PYTHONPATH=$PWD:$PWD/data_processing:$PYTHONPATH   # run from the repository root
```

All commands below run from the repository root in this environment.

## Datasets

### MegaDepth

```bash
python data_processing/wai_processing/download_scripts/download_megadepth.py /raw/megadepth_download
python -m wai_processing.scripts.conversion.megadepth original_root=/raw/megadepth_download/megadepth root=/data/wai/megadepth
python -m wai_processing.scripts.run_moge root=/data/wai/megadepth batch_size=1      # sky / far-background masks
python -m wai_processing.scripts.covisibility data_processing/wai_processing/configs/covisibility/covisibility_gt_depth.yaml root=/data/wai/megadepth
```

The download script fetches MegaDepth v1 and its SfM models, extracts and merges them, and adds the pair file.

### ETH3D

```bash
python data_processing/wai_processing/download_scripts/download_eth3d.py /raw/eth3d
python -m wai_processing.scripts.conversion.eth3d original_root=/raw/eth3d root=/data/wai/eth3d
python -m wai_processing.scripts.covisibility data_processing/wai_processing/configs/covisibility/covisibility_gt_depth.yaml root=/data/wai/eth3d
```

### KITTI

The KITTI odometry colour images, Velodyne scans, calibration and poses (154 GB), extracted into one folder
(`sequences/`, `poses/`). Sequences 00–08 are converted, in chunks of 100 frames; sequence 00 is the validation split.
A subset works too: only the frames on disk are converted.

```bash
mkdir -p /raw/kitti && for z in data_odometry_color data_odometry_velodyne data_odometry_calib data_odometry_poses; do
    wget -P /raw/kitti https://s3.eu-central-1.amazonaws.com/avg-kitti/$z.zip && unzip -q -d /raw/kitti /raw/kitti/$z.zip
done
python -m wai_processing.scripts.conversion.kitti original_root=/raw/kitti/dataset root=/data/wai/kitti
python -m wai_processing.scripts.covisibility data_processing/wai_processing/configs/covisibility/covisibility_point_hits.yaml root=/data/wai/kitti
```

### Oxford Spires

```bash
python -m wai_processing.scripts.conversion.download_spires --out /raw/oxford_spires
cd /raw/oxford_spires/sequences
for s in */; do unzip -qn $s/raw/images.zip -d $s/raw/images; unzip -qn $s/processed/vilens-slam/undist-clouds.zip -d $s/processed/vilens-slam; done
cd - && cp data_processing/wai_processing/assets/oxford_spires/sensor.yaml /raw/oxford_spires/sequences/
python -m wai_processing.scripts.conversion.osd original_root=/raw/oxford_spires/sequences root=/data/wai/osd
python -m wai_processing.scripts.covisibility data_processing/wai_processing/configs/covisibility/covisibility_point_hits.yaml root=/data/wai/osd
```

### Argoverse

[Argoverse 1](https://www.argoverse.org/av1.html) 3D tracking v1.1 (`tracking_train{1,2,3}_v1.1`), extracted into one folder.

```bash
python -m wai_processing.scripts.conversion.argoverse original_root=/raw/argoverse root=/data/wai/argoverse
python -m wai_processing.scripts.covisibility data_processing/wai_processing/configs/covisibility/covisibility_point_hits.yaml root=/data/wai/argoverse
```

### PandaSet

[PandaSet](https://pandaset.org) as released (`<scene>/camera/`, `<scene>/lidar/`).

```bash
python -m wai_processing.scripts.conversion.pandaset original_root=/raw/pandaset root=/data/wai/pandaset
python -m wai_processing.scripts.pandaset_covisibility --root /data/wai/pandaset
```

### nuScenes

Register at [nuscenes.org](https://www.nuscenes.org/nuscenes#download) and download from *Full dataset (v1.0)*:

| Split | Archives | Size |
|---|---|---|
| Trainval (training) | `v1.0-trainval_meta.tgz`, `v1.0-trainval01_blobs.tgz` ... `v1.0-trainval10_blobs.tgz` | ~335 GB |
| Test (benchmark) | `v1.0-test_meta.tgz`, `v1.0-test_blobs.tgz` | ~58 GB |

Every archive unpacks to the same top-level folders, so extracting a split's archives into one folder merges them:

```bash
DL=~/Downloads/nuscenes          # where the .tgz files are
RAW=/raw/nuscenes                 # where the extracted dataset goes
mkdir -p $RAW/trainval $RAW/test
for f in $DL/v1.0-trainval_meta.tgz $DL/v1.0-trainval{01..10}_blobs.tgz; do tar -xzf "$f" -C $RAW/trainval; done
for f in $DL/v1.0-test_meta.tgz $DL/v1.0-test_blobs.tgz; do tar -xzf "$f" -C $RAW/test; done
```

Afterwards each split looks like this (`test/` the same, with `v1.0-test/` in place of `v1.0-trainval/`):

```
/raw/nuscenes/trainval/
├── LICENSE
├── maps/                          # *.png
├── samples/                       # keyframes, one folder per sensor
│   ├── CAM_BACK/ ... CAM_FRONT_RIGHT/   # 6 cameras: n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915243012465.jpg
│   ├── LIDAR_TOP/                 # n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243097718.pcd.bin
│   └── RADAR_BACK_LEFT/ ... RADAR_FRONT_RIGHT/
├── sweeps/                        # the frames between keyframes, same sensor folders
└── v1.0-trainval/                 # metadata tables
    ├── attribute.json  calibrated_sensor.json  category.json  ego_pose.json  instance.json  log.json
    └── map.json  sample.json  sample_annotation.json  sample_data.json  scene.json  sensor.json  visibility.json
```

`nuscenes_preprocess.py` builds the image / point-cloud pairs (the CorrI2P / VP2P protocol). It needs the nuScenes
devkit, which pins old dependencies, so give it its own environment. The test set is what
`scripts/eval.py --dataset nuscenes` reads.

```bash
conda create -n nuscenes python=3.8 -y && conda activate nuscenes
pip install nuscenes-devkit opencv-python
python data_processing/nuscenes_preprocess.py --dataroot $RAW/trainval --out /data/nuscenes_processed/train
python data_processing/nuscenes_preprocess.py --dataroot $RAW/test --version v1.0-test --frame_skip 2 \
    --out /data/nuscenes_test_processed/test
conda activate drs_vpt_data
python -m wai_processing.scripts.conversion.nuscenes original_root=/data/nuscenes_processed root=/data/wai/nuscenes
python -m wai_processing.scripts.covisibility data_processing/wai_processing/configs/covisibility/covisibility_point_hits.yaml root=/data/wai/nuscenes hit_radius=0.5
```

### 7-Scenes

The [7-Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) RGB-D sequences, plus one map
per scene fused from its training sequences:

```bash
python data_processing/tsdf_fusion_7scenes.py --dataset_root /raw/7scenes
python -m wai_processing.scripts.conversion.7scenes original_root=/raw/7scenes root=/data/wai/7scenes
python -m wai_processing.scripts.covisibility data_processing/wai_processing/configs/covisibility/covisibility_point_hits.yaml root=/data/wai/7scenes hit_radius=0.02
```

### WildRGB-D

All 46 categories, preprocessed with DUSt3R's script (used from its own checkout):

```bash
git clone https://github.com/wildrgbd/wildrgbd && git clone https://github.com/naver/dust3r
TOOLS=$PWD
mkdir -p /raw/wildrgbd && cd /raw/wildrgbd && python $TOOLS/wildrgbd/download.py --cat all && cd $TOOLS
python dust3r/datasets_preprocess/preprocess_wildrgbd.py --wildrgbd_dir /raw/wildrgbd --output_dir /data/wildrgbd_processed
python -m wai_processing.scripts.conversion.wildrgbd original_root=/data/wildrgbd_processed root=/data/wai/wildrgbd
python -m wai_processing.scripts.covisibility data_processing/wai_processing/configs/covisibility/covisibility_gt_depth.yaml root=/data/wai/wildrgbd
```

### GrandTour

`grand_tour_prepare.py` downloads the Hesai scans and camera images of the GrandTour missions from the
[Hugging Face Hub](https://huggingface.co/datasets/leggedrobotics/grand_tour_dataset) (about 9 GB per mission, no
registration) and motion-compensates the scans with the poses and per-frame motion shipped in
`wai_processing/assets/grand_tour/`:

```bash
python -m wai_processing.scripts.conversion.grand_tour_prepare --download /raw/grand_tour_hub --out /data/grand_tour_prepared
python -m wai_processing.scripts.conversion.grand_tour original_root=/data/grand_tour_prepared root=/data/wai/grand_tour
python -m wai_processing.scripts.covisibility data_processing/wai_processing/configs/covisibility/covisibility_point_hits.yaml root=/data/wai/grand_tour
```

## Scene lists

```bash
python data_processing/aggregate_scene_names.py --wai_root /data/wai --output_dir /data/wai_meta
```

It covers every dataset folder under `--wai_root` (or those given with `--datasets`), and writes
`<output_dir>/<split>/<dataset>_scene_list_<split>.npy` with the train / val / test splits of
`drs_vpt/_vendor/mapanything/datasets/utils/data_splits.py`.

## Provenance

`wai_processing/` is MapAnything's WAI processing suite (Apache 2.0), reduced to the stages above, plus the DRS VPT
converters. Changes to MapAnything's files are in `wai_processing/wai_processing.patch`.
