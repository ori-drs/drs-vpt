"""Load DRS VPT weights and demo samples from the Hugging Face Hub (or local copies).

Both are downloaded on first use and cached in the standard HF cache (~/.cache/huggingface,
override with HF_HOME / HF_HUB_CACHE). Private repos need `hf auth login` or HF_TOKEN.

    model = load_model(device="cuda")   # fulkast/drs_vpt
    sample = load_sample("kitti")       # fulkast/drs_vpt_demo_data/kitti, or a local folder

A sample is one map cloud and one or more query frames:
    sample["cloud"] (N x 3 float32, metres), sample["grid_size"], sample["frames"] = [{"img_path", "K", "T_c2w"}]
plus the fields of its meta.json (title, source, licence, optional "up" axis).
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

MODEL_REPO = "fulkast/drs_vpt"
DATA_REPO = "fulkast/drs_vpt_demo_data"
REVISION = "v1.0"  # model weights
DATA_REVISION = "v1.1"  # demo samples (v1.0) plus the benchmark regression set

HUB_SAMPLES = ("kitti", "7scenes", "12scenes", "iphone", "iphone_stream")

def resolve_special_float(value):
    """OmegaConf resolver for "inf" / "-inf" in the model config."""
    if value == "inf":
        return np.inf
    if value == "-inf":
        return -np.inf
    raise ValueError(f"Unknown special float value: {value}")


def load_model(checkpoint: Optional[str] = None, device="cpu", revision: Optional[str] = REVISION,
               cloud_device=None):
    """Return DRSVPT in eval mode on `device`.

    checkpoint: None for the Hub weights, a Hub repo id, a local directory written by
    save_pretrained, or a local training checkpoint (.pth with a "model" state dict).
    cloud_device: where the Sonata cloud encoder runs. Defaults to `device`, except on MPS, where spconv has no
    kernels and it defaults to the CPU; the model moves the encoded cloud to `device` itself.
    """
    import spconv.pytorch as spconv

    from drs_vpt.model import DRSVPT

    checkpoint = checkpoint or MODEL_REPO
    if Path(checkpoint).is_file():
        model = _load_training_checkpoint(checkpoint)
    else:
        model = DRSVPT.from_pretrained(checkpoint, revision=revision, strict=True)
    model = model.eval().float()

    # Sonata's default spconv algorithm autotunes on its first call (~4 s); Native needs no tuning
    # and is as fast at inference.
    for m in model.modules():
        if isinstance(m, spconv.conv.SparseConvolution):
            m.algo = spconv.ConvAlgo.Native

    if cloud_device is None:
        cloud_device = "cpu" if torch.device(device).type == "mps" else device
    model.cloud_encoder.to(cloud_device)

    skip = {id(m) for m in model.cloud_encoder.modules()}
    for m in model.modules():
        if id(m) not in skip:
            for k, p in m._parameters.items():
                if p is not None:
                    p.data = p.data.to(device)
            for k, b in m._buffers.items():
                if b is not None:
                    m._buffers[k] = b.to(device)
    return model


def _load_training_checkpoint(path: str):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from drs_vpt.model import DRSVPT

    if not OmegaConf.has_resolver("special_float"):
        OmegaConf.register_new_resolver("special_float", resolve_special_float)
    config_dir = str(Path(__file__).parents[1] / "configs")  # the training configs at the repo root
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="model/drs-vpt")
    cfg = OmegaConf.to_container(cfg, resolve=True)
    model = DRSVPT(**cfg["model"]["model_config"], skip_encoder_pretrain=True)
    state_dict = torch.load(path, map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(state_dict, strict=True)
    return model


def _sample_dir(name_or_dir: str, revision: Optional[str]) -> Path:
    d = Path(name_or_dir)
    if d.is_dir():
        return d
    if name_or_dir not in HUB_SAMPLES:
        raise ValueError(f"{name_or_dir!r} is neither a local folder nor a Hub sample {HUB_SAMPLES}")
    from huggingface_hub import snapshot_download
    root = snapshot_download(DATA_REPO, repo_type="dataset", revision=revision, allow_patterns=f"{name_or_dir}/**")
    return Path(root) / name_or_dir


def regression_data_dir(revision: Optional[str] = DATA_REVISION) -> Path:
    """The benchmark regression set, <dir>/{kitti,nuscenes}/<name>/ (written by scripts/export_regression_set.py),
    downloaded from the demo data repo (about 60 MB) and cached."""
    from huggingface_hub import snapshot_download
    root = snapshot_download(DATA_REPO, repo_type="dataset", revision=revision, allow_patterns="regression/**")
    return Path(root) / "regression"


def load_sample(name_or_dir: str, revision: Optional[str] = DATA_REVISION) -> dict:
    """Load a sample: a Hub sample name (downloaded and cached) or a local folder.

    Two folder layouts are read:
      one frame:     image.{png,jpg}, K.npy, cloud.npy, meta.json (grid_size, optional T_c2w, or P = its inverse)
      many frames:   cloud.npy, frames/*, meta.json (grid_size, "frames": [{"image", "K", optional "T_c2w"}])
    Either way: meta.json's fields, cloud, and frames = [{"image", "img_path", "K", "T_c2w" (or None)}].
    """
    d = _sample_dir(name_or_dir, revision)
    meta = json.loads((d / "meta.json").read_text())
    if "frames" in meta:
        frames = [{**f, "img_path": str(d / f["image"]), "K": np.array(f["K"]),
                   "T_c2w": np.array(f["T_c2w"]) if f.get("T_c2w") is not None else None}
                  for f in meta.pop("frames")]
    else:
        img = next(d.glob("image.*"))
        T = meta.pop("T_c2w", None)
        if T is None and "P" in meta:  # the KITTI samples store the ground truth as P = T_w2c
            T = np.linalg.inv(np.array(meta["P"]))
        frames = [{"image": img.name, "img_path": str(img), "K": np.load(d / "K.npy"),
                   "T_c2w": np.array(T) if T is not None else None}]
    return {"title": d.name, **meta, "frames": frames, "cloud": np.load(d / "cloud.npy")}
