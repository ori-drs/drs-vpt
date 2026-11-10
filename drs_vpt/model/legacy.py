# Copyright (C) 2024-present Oxford Robotics Institute.
# Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

"""Converting checkpoints from before the module rename into the current DRSVPT format.

Older checkpoints (fulkast/drs_vpt before its October 2026 re-export, and training checkpoints from that time) use the
model's development name for the alignment modules (tamaya_* instead of align_*, name "drs-vpt-tamaya"), and their weights hold
geometric-conditioning encoders that DRS VPT never used. Nothing imports this module; use it to convert such a
checkpoint once, then load the result as usual:

    from drs_vpt.model import legacy
    ckpt = torch.load("old.pth", map_location="cpu", weights_only=False)
    ckpt["model"] = legacy.upgrade_legacy_state_dict(ckpt["model"])
    torch.save(ckpt, "converted.pth")   # loads with drs_vpt.hub.load_model("converted.pth")

For an old Hub-format directory (config.json + model.safetensors), convert config.json with upgrade_legacy_config and
model.safetensors with upgrade_legacy_state_dict.
"""

LEGACY_NAME = "tamaya"
LEGACY_STRINGS = {"drs-vpt-tamaya": "drs-vpt", "tamaya_24": "align_feat_24"}

# Encoders for geometric conditioning (ray directions, depth, camera pose) that older checkpoints hold.
# DRS VPT was trained with every conditioning probability at 0, so they were never used and are not built.
UNUSED_CHECKPOINT_PREFIXES = ("ray_dirs_encoder.", "depth_encoder.", "depth_scale_encoder.", "cam_rot_encoder.",
                              "cam_trans_encoder.", "cam_trans_scale_encoder.")


def upgrade_legacy_config(cfg):
    """Rename legacy keys (tamaya_* -> align_*) and names (e.g. the model name) in a config dict, in place."""
    if not isinstance(cfg, dict):
        return cfg
    for key in list(cfg):
        value = upgrade_legacy_config(cfg.pop(key))
        if isinstance(value, str):
            value = LEGACY_STRINGS.get(value, value)
        cfg[key.replace(f"{LEGACY_NAME}_", "align_", 1) if key.startswith(f"{LEGACY_NAME}_") else key] = value
    return cfg


def upgrade_legacy_state_dict(state_dict):
    """Return the state dict with legacy keys renamed (tamaya_* -> align_*) and the unused encoders dropped."""
    return {
        (f"align_{k[len(LEGACY_NAME) + 1:]}" if k.startswith(f"{LEGACY_NAME}_") else k): v
        for k, v in state_dict.items()
        if not k.startswith(UNUSED_CHECKPOINT_PREFIXES)
    }
