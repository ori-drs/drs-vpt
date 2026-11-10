#!/usr/bin/env python3
"""DRS VPT demo: localize images inside a point cloud map.

A sample is one map cloud and one or more query frames (see drs_vpt.hub.load_sample for the folder layouts). The
model and the Hub samples (kitti, 7scenes, 12scenes, iphone: one frame each; iphone_stream: many frames) download
on first use. scripts/export_demo_sample.py writes samples from your own data or from benchmark datasets.

The map is encoded once; on MPS its encoder (Sonata) runs on the CPU, since spconv has no MPS kernels. Each frame
then runs only the image encoder, the fusion transformer, the heads and the aligner. Every frame is localized on its
own: nothing carries over from the previous frame.

    python scripts/demo.py                                  # KITTI sample
    python scripts/demo.py --sample 7scenes --viz3d         # one frame: overlay + 3-D view with frusta
    python scripts/demo.py --sample iphone_stream --every 4 # many frames: per-frame overlays + trajectory
    python scripts/demo.py --sample path/to/folder          # your own sample
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # avoids OOM from fragmentation on small GPUs

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from drs_vpt.hub import load_model, load_sample
from drs_vpt.inference import encode_map, localize, pick_autocast, read_image, resize_to_model
from drs_vpt.viz import (pose_error, rotate_view, save_3d_view, save_overlay_figure, save_trajectory_view,
                         upright_rotation)


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", default="kitti",
                   help="Hub sample (kitti, 7scenes, 12scenes, iphone, iphone_stream) or a local folder (default: kitti)")
    p.add_argument("--checkpoint", default=None, help="Local save_pretrained dir or training .pth instead of the Hub model")
    p.add_argument("--device", default=default_device(), help="cuda, mps or cpu (default: the fastest available)")
    p.add_argument("--max-points", type=int, default=150_000,
                   help="Subsample the map to at most this many points before voxelization (0 = all). "
                        "Lower it if you run out of memory.")
    p.add_argument("--every", type=int, default=1, help="Localize every n-th frame")
    p.add_argument("--max-frames", type=int, default=0, help="Stop after this many frames (0 = all)")
    p.add_argument("--predict-intrinsics", action="store_true",
                   help="Do not give the model each frame's K; use the intrinsics recovered from its ray map")
    p.add_argument("--no-overlays", action="store_true", help="Skip the per-frame overlay figures")
    p.add_argument("--viz3d", action="store_true",
                   help="One frame: also save the image above a 3-D view of the cloud and frusta")
    p.add_argument("--viz3d-show", action="store_true", help="Open the 3-D / trajectory view interactively")
    p.add_argument("--output", default="demo_output", help="Directory for the figures")
    return p.parse_args()


def main():
    args = parse_args()

    # 1. The model (fulkast/drs_vpt), downloaded and cached on first use. On MPS the cloud encoder stays on the CPU.
    model = load_model(args.checkpoint, device=args.device)

    # 2. A sample: one map cloud and its query frames (+ ground truth, if known)
    sample = load_sample(args.sample)
    frames = sample["frames"][::args.every]
    if args.max_frames > 0:
        frames = frames[:args.max_frames]
    print(f"Sample  : {sample['title']} ({len(frames)} of {len(sample['frames'])} frames, "
          f"{len(sample['cloud'])} map points)")
    if sample.get("source"):
        print(f"Source  : {sample['source']}")

    # 3. Encode the map once
    t0 = time.perf_counter()
    map_encoding = encode_map(model, sample["cloud"], sample["grid_size"], max_points=args.max_points)
    print(f"Map     : encoded in {time.perf_counter() - t0:.1f} s")

    # 4. Localize each frame: the camera pose in the cloud frame, world-to-camera (x_cam = R @ x_cloud + t)
    autocast, precision = pick_autocast(args.device)
    print(f"Device  : {args.device}, {precision}")
    out_dir = Path(args.output) / Path(args.sample).name
    single = len(frames) == 1
    results, times, errors = [], [], []
    for n, frame in enumerate(frames):
        image = read_image(frame["img_path"])
        t0 = time.perf_counter()
        out = localize(model, image, map_encoding, None if args.predict_intrinsics else frame["K"], autocast)
        times.append(time.perf_counter() - t0)

        H, W = out["image_shape"]
        T_gt = np.linalg.inv(frame["T_c2w"]) if frame["T_c2w"] is not None else None
        results.append({"T_w2c": out["T_w2c"], "T_w2c_coarse": out["T_w2c_coarse"], "K": out["K"],
                        "image_shape": (H, W), "T_w2c_gt": T_gt})
        line = f"[{n + 1:3d}/{len(frames)}] {Path(frame['image']).name}  {times[-1]:5.2f} s"
        if T_gt is not None:
            errors.append(pose_error(out["T_w2c"], frame["T_c2w"]))
            line += f"  GT error {errors[-1][0]:5.2f} deg, {errors[-1][1]:.3f} m"
        print(line, flush=True)
        if single:
            print(f"T_w2c   :\n{np.array2string(out['T_w2c'], precision=4, suppress_small=True)}")
        if args.predict_intrinsics:
            K_given = resize_to_model(image, frame["K"])[1]  # the frame's K at the model's resolution
            fmt = lambda k: f"fx {k[0, 0]:.1f}  fy {k[1, 1]:.1f}  cx {k[0, 2]:.1f}  cy {k[1, 2]:.1f}"
            print(f"          K predicted {fmt(out['K'])} | given {fmt(K_given)}")

        # Figures: the map projected into the frame with each pose (upright when meta.json gives the up axis)
        image = resize_to_model(image)[0]  # the image the model saw, which out["K"] refers to
        poses = {"Raw model output (transformer)": out["T_w2c_coarse"], "Fine alignment (c2f level 2)": out["T_w2c"]}
        if single:
            poses = {"Cloud input (identity pose)": np.eye(4, dtype=np.float32), **poses}
        if T_gt is not None:
            poses["Ground truth"] = T_gt
        if not args.no_overlays:
            T_up = T_gt if T_gt is not None else out["T_w2c"]  # GT is right even when localization fails
            im, K, ps = rotate_view(image, out["K"], poses, upright_rotation(T_up, sample.get("up")))
            name = f"{Path(args.sample).name}_demo.png" if single else f"{Path(frame['image']).stem}.png"
            save_overlay_figure(im, sample["cloud"], K, ps, f"{sample['title']}: {Path(frame['image']).name}",
                                out_dir / name)
        if single and (args.viz3d or args.viz3d_show):
            save_3d_view(sample["cloud"], out["K"], image, poses, sample["title"],
                         out_dir / f"{Path(args.sample).name}_demo_3d.png", interactive=args.viz3d_show,
                         up=sample.get("up"))

    # 5. Summary and, for several frames, the trajectory in the map
    if not single:
        steady = times[1:] or times  # the first frame includes one-off warm-up
        print(f"Time    : {np.median(steady):.2f} s/frame median ({1 / np.median(steady):.2f} frames/s), "
              f"first frame {times[0]:.2f} s")
        if errors:
            rot, trans = np.array(errors).T
            print(f"GT error: median {np.median(rot):.2f} deg, {np.median(trans):.3f} m; "
                  f"{np.mean((rot < 5) & (trans < 0.05)) * 100:.0f}% of frames within 5 deg / 5 cm")
        save_trajectory_view(sample["cloud"], results, sample["title"], out_dir / "trajectory_3d.png",
                             up=sample.get("up"), interactive=args.viz3d_show)


if __name__ == "__main__":
    main()
