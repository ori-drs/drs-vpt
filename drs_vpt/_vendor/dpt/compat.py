# Written for DRS VPT (Oxford Robotics Institute), Apache License 2.0.
#
# The helpers UniCeption's DPT head (prediction_heads/dpt.py) imports. UniCeption takes them from its copy of CroCo's
# dpt_block.py (Naver, CC BY-NC-SA 4.0); here they are built on DPT's own MIT blocks (blocks.py) instead. With the
# settings DRS VPT uses (width_ratio=1, ReLU) the modules, their parameter names and their outputs are the same.
# CroCo's non-square upsampling (width_ratio != 1) and UniCeption's custom activations (sine, gaussian, xcosx) are not
# reproduced.

import torch.nn as nn

from drs_vpt._vendor.dpt.blocks import FeatureFusionBlock_custom, _make_scratch


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def make_scratch(in_shape, out_shape, groups=1, expand=False):
    """DPT's scratch module, plus layer_rn: the four projection convs as a list (the DPT head pairs each with its
    post-processing layer)."""
    scratch = _make_scratch(in_shape, out_shape, groups=groups, expand=expand)
    scratch.layer_rn = nn.ModuleList([scratch.layer1_rn, scratch.layer2_rn, scratch.layer3_rn, scratch.layer4_rn])
    return scratch


_NONLINEARITIES = {"relu": lambda: nn.ReLU(False), "tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "gelu": nn.GELU}


def make_nonlinearity(nonlinearity, dim=None, on_channels=False):
    if nonlinearity not in _NONLINEARITIES:
        raise ValueError(f"Unsupported nonlinearity: {nonlinearity} (DRS VPT supports {sorted(_NONLINEARITIES)})")
    return _NONLINEARITIES[nonlinearity]()


def make_fusion_block(features, use_bn, width_ratio=1, nonlinearity="relu"):
    if width_ratio != 1:
        raise ValueError("Non-square DPT upsampling (width_ratio != 1) is not supported")
    return FeatureFusionBlock_custom(features, make_nonlinearity(nonlinearity), deconv=False, bn=use_bn, expand=False,
                                     align_corners=True)
