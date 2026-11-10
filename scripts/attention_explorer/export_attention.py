#!/usr/bin/env python3
"""Export one sample's attention and features for the attention explorer (attention/ in the ori-drs/drs-vpt-pages repo).

Runs the model on a sample (a Hub sample name or a folder, see drs_vpt.hub.load_sample) and records, in each of the
12 global blocks of the image-cloud transformer, the attention of every image patch over the cloud tokens and the
cloud and image token features after the block.

    python scripts/attention_explorer/export_attention.py --sample 7scenes --out ../drs-vpt-pages/attention/data/stairs
    python scripts/attention_explorer/export_attention.py --sample path/to/megadepth_sample --out ... --max-points 38912
    python scripts/attention_explorer/export_attention.py --sample path/to/sample --image photo.jpg --predict-k --out ...

Writes to --out:
  image.jpg   the image at the model's resolution (what the patches index)
  cloud.bin   uint16 (N, 3) display points quantized in the box meta["bbox"], then uint16 (N,) token per point
  attn.bin    per global block: uint16 (P, topk) token indices, then uint8 (P, topk) values v, where
              log2(weight * n_tokens) = v / 255 * (log_hi - log_lo) + log_lo (weights are head means over the cloud tokens)
  feat.bin    per global block: uint8 (n_tokens, 3) PCA colours of the cloud token features, then uint8 (P, 3) the image
              patches projected onto the same components
  meta.json   shapes, predicted pose, intrinsics, per-block summary numbers
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from drs_vpt.hub import load_model, load_sample
from drs_vpt.inference import _image_view, encode_map, localize, read_image
from drs_vpt.viz import pose_error

PATCH = 14
LOG_LO, LOG_HI = -1.0, 7.0


def capture_blocks(model, state, captured):
    """Hook the global blocks (even indices: image, cloud and scale tokens attend to each other)."""
    blocks = model.info_sharing.self_attention_blocks
    global_ids = list(range(0, len(blocks), 2))
    for i in global_ids:
        attn = blocks[i].attn
        assert attn.custom_positional_encoding is None
        assert not getattr(attn, "use_scalable_softmax", False) and not getattr(attn, "use_entropy_scaling", False)

        def attention(x, xpos=None, attn=attn, forward=attn.forward, i=i):
            # recompute softmax(q k^T) for the image rows only; the block's own (fused) forward runs unchanged
            P, T = state["P"], state["T"]
            B, N, _ = x.shape
            qkv = attn.qkv(x).reshape(B, N, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
            q, k = attn.q_norm(qkv[0]), attn.k_norm(qkv[1])
            a = ((q[:, :, :P] * attn.scale) @ k.transpose(-2, -1)).softmax(-1)
            cloud = a[0, :, :, P:P + T].mean(0)  # head mean, (P, T)
            captured[i] = {"cloud_share": cloud.sum(-1).numpy(), "w": (cloud / cloud.sum(-1, keepdim=True)).numpy()}
            return forward(x, xpos)

        def features(_module, _inp, out, i=i):
            P, T = state["P"], state["T"]
            captured[i]["img_feat"] = out[0, :P].float().numpy()
            captured[i]["cloud_feat"] = out[0, P:P + T].float().numpy()

        attn.forward = attention
        blocks[i].register_forward_hook(features)
    return global_ids


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", required=True, help="Hub sample name or sample folder (first frame is used)")
    p.add_argument("--out", required=True, help="Output folder, e.g. ../drs-vpt-pages/attention/data/<scene>")
    p.add_argument("--image", help="Query image to use instead of the sample's (no ground-truth pose then)")
    p.add_argument("--predict-k", action="store_true", help="Give the model no intrinsics; it predicts them")
    p.add_argument("--max-points", type=int, default=0, help="Map points the model sees (0 = all)")
    p.add_argument("--display-points", type=int, default=1_000_000, help="Map points the page shows")
    p.add_argument("--topk", type=int, default=160, help="Cloud tokens kept per patch and block")
    args = p.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(device="cpu")
    model.eval()
    sample = load_sample(args.sample)
    frame = sample["frames"][0]
    if args.image:
        frame = {"img_path": args.image, "K": None, "T_c2w": None}
        sample["source"] = sample.get("source", "") + f"; query image: {Path(args.image).name}"
    if args.predict_k:
        frame["K"] = None
    image = read_image(frame["img_path"])
    state, captured = {}, {}
    global_ids = capture_blocks(model, state, captured)

    menc = encode_map(model, sample["cloud"], grid_size=sample["grid_size"], max_points=args.max_points)
    views = _image_view(model, image, frame["K"])
    H, W = (int(s) for s in views[0]["img"].shape[-2:])
    gh, gw = H // PATCH, W // PATCH
    state["P"], state["T"] = gh * gw, menc["feats"].shape[1]
    out = localize(model, image, menc, frame["K"])
    T_w2c, K = out["T_w2c"], out["K"]
    err = pose_error(T_w2c, frame["T_c2w"]) if frame["T_c2w"] is not None else None
    print(f"image {H}x{W}, {gh}x{gw} patches, {state['T']} tokens, pose error {err}")

    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])  # dinov2 normalization
    img = views[0]["img"][0].permute(1, 2, 0).numpy() * std + mean
    Image.fromarray((img.clip(0, 1) * 255).astype(np.uint8)).save(out_dir / "image.jpg", quality=90)

    # the cloud encoder works in coordinates divided by the scene scale (log1p of the mean point distance); undo it
    scene_scale = float(menc["scene_scale_per_point"][0])
    voxel_xyz = menc["cloud_view"].coord.float().numpy() * scene_scale
    voxel_tok = menc["inv_gather"].numpy()
    coord = sample["cloud"].astype(np.float32)
    if len(coord) > args.display_points:
        coord = coord[np.random.default_rng(1).choice(len(coord), args.display_points, replace=False)]
    tok_of_point = voxel_tok[cKDTree(voxel_xyz).query(coord)[1]].astype(np.uint16)  # nearest voxel's token
    lo, hi = coord.min(0), coord.max(0)
    with open(out_dir / "cloud.bin", "wb") as f:
        f.write(np.round((coord - lo) / np.maximum(hi - lo, 1e-9) * 65535).astype(np.uint16).tobytes())
        f.write(tok_of_point.tobytes())

    # token centres projected with the predicted pose, for the per-block locality numbers
    n_tok = state["T"]
    cnt = np.bincount(tok_of_point, minlength=n_tok)
    tok_xyz = np.stack([np.bincount(tok_of_point, coord[:, j], n_tok) for j in range(3)], 1) / np.maximum(cnt, 1)[:, None]
    tc = tok_xyz @ T_w2c[:3, :3].T + T_w2c[:3, 3]
    tuv = (tc @ K.T)[:, :2] / np.maximum(tc[:, 2], 1e-6)[:, None]
    tvis = (tc[:, 2] > 0) & (tuv[:, 0] >= 0) & (tuv[:, 0] < W) & (tuv[:, 1] >= 0) & (tuv[:, 1] < H)
    py, px = np.mgrid[0:gh, 0:gw]
    centres = np.stack([(px.ravel() + .5) * PATCH, (py.ravel() + .5) * PATCH], 1)
    dist = np.linalg.norm(centres[:, None, :] - tuv[None, tvis, :], axis=-1)

    layers_meta = []
    with open(out_dir / "attn.bin", "wb") as f:
        for i in global_ids:
            w = captured[i]["w"]
            idx = np.argsort(-w, axis=1)[:, :args.topk]
            top = np.take_along_axis(w, idx, 1)
            q = np.round((np.clip(np.log2(top * n_tok), LOG_LO, LOG_HI) - LOG_LO) / (LOG_HI - LOG_LO) * 255)
            f.write(idx.astype(np.uint16).tobytes())
            f.write(q.astype(np.uint8).tobytes())
            wv = w[:, tvis] / w[:, tvis].sum(1, keepdims=True)
            layers_meta.append({
                "block": i,
                "cloud_share": float(captured[i]["cloud_share"].mean()),  # attention mass on cloud tokens
                "top50_share": float(np.sort(w, 1)[:, -50:].sum(1).mean()),  # mass on the 50 strongest tokens
                "entropy_ratio": float((-(w * np.log(w + 1e-12)).sum(1)).mean() / np.log(n_tok)),  # 1 = uniform
                "px_dist": float((wv * dist).sum(1).mean()),  # attention-weighted reprojection distance
            })
            print(layers_meta[-1])

    with open(out_dir / "feat.bin", "wb") as f:
        for i in global_ids:
            cf, imf = captured[i]["cloud_feat"], captured[i]["img_feat"]
            mu = cf.mean(0)
            basis = np.linalg.svd(cf - mu, full_matrices=False)[2][:3].T
            pc, pi = (cf - mu) @ basis, (imf - mu) @ basis
            lo_q, hi_q = np.percentile(pc, 2, axis=0), np.percentile(pc, 98, axis=0)
            for x in (pc, pi):
                f.write((np.clip((x - lo_q) / (hi_q - lo_q), 0, 1) * 255).round().astype(np.uint8).tobytes())

    # where each patch lands under the predicted pose, for the default hover patch
    pcam = coord @ T_w2c[:3, :3].T + T_w2c[:3, 3]
    uv = (pcam @ K.T)[:, :2] / np.maximum(pcam[:, 2], 1e-6)[:, None]
    vis = (pcam[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < gw * PATCH) & (uv[:, 1] >= 0) & (uv[:, 1] < gh * PATCH)
    patch_of_point = (uv[vis, 1] // PATCH).astype(int) * gw + (uv[vis, 0] // PATCH).astype(int)
    has_map = np.bincount(patch_of_point, minlength=gh * gw) >= 8
    last = captured[global_ids[-1]]["w"][:, tvis]
    local = (last / last.sum(1, keepdims=True) * dist).sum(1)
    centre_w = np.exp(-(((px.ravel() + .5) / gw - .5) ** 2 + ((py.ravel() + .5) / gh - .5) ** 2) * 6)
    default_patch = int(np.argmin(np.where(has_map, local / centre_w, np.inf)))

    meta = {
        "sample": Path(args.sample).name, "title": sample.get("title", args.sample), "source": sample.get("source", ""),
        "license": sample.get("license", ""), "H": H, "W": W, "gh": gh, "gw": gw, "patch": PATCH,
        "n_points": int(len(coord)), "n_voxels": int(len(voxel_xyz)), "scene_scale": scene_scale,
        "max_points": args.max_points, "bbox": [lo.tolist(), hi.tolist()], "n_tokens": int(n_tok), "topk": args.topk,
        "log_lo": LOG_LO, "log_hi": LOG_HI, "T_w2c": T_w2c.tolist(), "K": K.tolist(), "pose_error": err,
        "default_patch": default_patch, "layers": layers_meta,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
