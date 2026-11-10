# DRS VPT: quickstart

## What you get

```bash
python scripts/demo.py                    # KITTI sample
python scripts/demo.py --sample iphone    # or 7scenes, 12scenes, or a local sample folder
python scripts/demo.py --predict-intrinsics  # ignore K.npy; use K recovered from the predicted ray map
python scripts/demo.py --sample iphone_stream  # stream: encode the map once, then localize many frames in it
```

- The model `fulkast/drs_vpt` (2.9 GB) and the samples `fulkast/drs_vpt_demo_data` download from the Hugging Face Hub
  on first use. They are cached in `~/.cache/huggingface`; set `HF_HOME` to move the cache. Nothing else is fetched:
  DINOv2 is built from the vendored code and the Sonata config is stored in the model's `config.json`.

The demo prints the pose `T_w2c` (world-to-camera; `x_cam = R @ x_cloud + t`), the GT error when the sample has a
ground-truth pose, and saves `demo_output/<sample>_demo.png` (`--viz3d` adds a 3-D view).

## Code map

| File | Role |
|---|---|
| `scripts/demo.py` | The demo, five commented steps. Start here. |
| `drs_vpt/hub.py` | `load_model(checkpoint=None, device=...)` (Hub, `save_pretrained` dir, or training `.pth`) and `load_sample(name_or_dir)` |
| `drs_vpt/inference.py` | `read_image`, `encode_map`, `localize(model, image, map_encoding, K)` (returns 4x4 `T_w2c`, `T_w2c_coarse`, the `K` it used, `raw`; `K=None` predicts the intrinsics), `resize_to_model`, `calibrate` (camera-LiDAR extrinsic) |
| `drs_vpt/viz.py` | Overlay figure, 3-D PyVista view, `pose_error` |
| `drs_vpt/eval.py`, `scripts/eval.py` | KITTI and nuScenes benchmark evaluation (see README.md). |
| `scripts/export_demo_sample.py` | All dataset-specific loading. It turns a frame from KITTI / nuScenes / 7Scenes / 12Scenes / Cambridge / RIO10 / a Polycam iPhone scan / custom files into a sample folder. The Hub samples were made with it. |
| `tests/` | `pytest tests` (uses the Hub model and samples). Fixtures are in `tests/assets/regression/`; regenerate them with `python tests/record_regression.py`. They also check 10 KITTI and 10 nuScenes benchmark samples (`regression/` in `fulkast/drs_vpt_demo_data`, about 60 MB); `VPT_REGRESSION_DATA` points them at a local set exported by `scripts/export_regression_set.py` instead. |

**Sample folder format:**
- `image.{png,jpg}`
- `K.npy`: 3x3 intrinsics of that image.
- `cloud.npy`: (N, 3) float32 map in metres.
- `meta.json`: `grid_size`, which is the voxel size: about 0.02 outdoors, 0.005–0.01 indoors. Optional `T_c2w` holds the ground-truth camera-to-world pose in the cloud frame. Optional `up` is the cloud's up axis (e.g. `[0, 1, 0]` for Polycam, whose ARKit world is y-up); `--viz3d` uses it to show sideways-stored portrait photos upright.

**Your own iPhone scan:** use a Polycam raw export:
```bash
python scripts/export_demo_sample.py --dataset polycam --data /path/to/export --out my_room
python scripts/demo.py --sample my_room
```

## Install

See the install blocks in `README.md`. If the macOS build fails, run the Linux CPU install in a container (on Apple
Silicon, enable Docker Desktop's "Use Rosetta for x86_64/amd64 emulation"):
```bash
docker run --platform linux/amd64 -it -v "$PWD":/work -w /work python:3.12 bash
pip install torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cpu
pip install spconv-cu126 torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
pip install -e ".[viz,dev]"
python scripts/demo.py --device cpu
```

## Hardware notes

- **CPU** works everywhere spconv installs. Inference runs in fp32 and takes 17–35 s per sample on a laptop CPU.
- **GPU**: the weights take 2.7 GiB (fp32). On a 4 GB card, lower `--max-points` (e.g. `--max-points 40000`) for the
  KITTI and 12scenes samples. fp16 autocast is chosen automatically on GPUs where it is faster.
- Keep Sonata's `enable_flash=False`: the model uses the fp32 attention path.

## Predicted intrinsics

The encoder never receives intrinsics, so the coarse pose and the predicted ray map come from the image alone; K is
only used by the coarse-to-fine aligner. With `--predict-intrinsics` (or `localize(..., K=None)`) the aligner uses the
pinhole K recovered from the predicted ray map instead.

## Localizing a stream (encode the map once)

`inference.encode_map(model, cloud, grid_size)` runs the image-independent half of the forward pass (Sonata and the
5k map tokens) once, and `localize(model, image, map_encoding, K)` reuses it for each frame. Each frame is
localized on its own; the model takes no pose prior.

`load_model(device="mps")` keeps the cloud encoder on the CPU and runs everything else on the Apple GPU.

Stream sample format: `cloud.npy`, `frames/*.jpg`, and `meta.json` with `"frames": [{"image", "K", "T_c2w"}]`.
`export_demo_sample.py --dataset polycam --stream` writes one from a Polycam export.

## Licence

- Code: Apache 2.0 (`LICENSE`). Our files carry an Oxford Robotics Institute header.
- Vendored code keeps its upstream licence, with a `LICENSE` file in each folder. Our modifications are noted in the
  file headers, and the exact diffs are in the `*.patch` files.
  - `drs_vpt/_vendor/`: dinov2, mapanything, pixloc and sonata are Apache 2.0; dpt is MIT; uniception is BSD 3-Clause.
  - `third_party/`: spconv and cumm are Apache 2.0; ccimport is MIT.
- Model weights (`fulkast/drs_vpt`): CC BY-NC 4.0 (non-commercial use only).

## Training

See `train.md`.
