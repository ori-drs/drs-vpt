# Copyright (C) 2024-present Oxford Robotics Institute.
# Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

import copy
import os
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from torch.utils.checkpoint import checkpoint
from drs_vpt._vendor.uniception.models.encoders import (
    encoder_factory,
    ViTEncoderInput,
)
from drs_vpt._vendor.uniception.models.info_sharing.base import MultiViewTransformerOutput
from drs_vpt._vendor.uniception.models.prediction_heads.adaptors import (
    CamTranslationPlusQuatsAdaptor,
    RayDirectionsPlusDepthWithConfidenceAndMaskAdaptor,
    ScaleAdaptor,
)
from drs_vpt._vendor.uniception.models.prediction_heads.base import (
    AdaptorInput,
    PredictionHeadInput,
    PredictionHeadLayeredInput,
    PredictionHeadTokenInput,
)
from drs_vpt._vendor.uniception.models.prediction_heads.dpt import DPTFeature, DPTRegressionProcessor
from drs_vpt._vendor.uniception.models.prediction_heads.mlp_head import MLPHead
from drs_vpt._vendor.uniception.models.prediction_heads.pose_head import PoseHead

from drs_vpt.model.adaptors import AlignFeatureWithConfidenceAdaptor
from drs_vpt.model.aggregation import (ZeroInitResidualFuse,
                                                                   dense_pack_simple,
                                                                   sample_evenly_with_inverse,
                                                                   gather_valid_indices)
from drs_vpt._vendor.pixloc.wrappers import Pose, Camera
from drs_vpt._vendor.pixloc.learned_optimizer import LearnedOptimizer
from drs_vpt.model.image_cloud_transformer import (
    MultiViewAlternatingAttentionTransformerIFR,
)
from drs_vpt.model.image_cloud_transformer import ImageCloudTransformerInput
from drs_vpt.model.wrapped_heads import make_wrapped_conv1x1
from drs_vpt._vendor import sonata
import spconv.pytorch as spconv
from drs_vpt._vendor.mapanything.utils.geometry import (
    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap, quaternion_to_rotation_matrix,
    recover_pinhole_intrinsics_from_ray_directions,
)
from drs_vpt._vendor.mapanything.utils.geometry import rotation_matrix_to_quaternion


def _spconv_forward_cpu_compat(self, input, add_input=None):
    """Make spconv layers run on CPU. Its default algo (implicit GEMM) is CUDA-only, and its
    inference kernels fuse the bias, which CPU rejects ("bias and act don't support cpu"); so on
    CPU use the Native algo and add the bias after the conv. CUDA inputs are untouched."""
    if input.features.is_cuda:
        return _spconv_forward_orig(self, input, add_input)
    algo, self.algo = self.algo, spconv.ConvAlgo.Native
    try:
        out = self._conv_forward(self.training, input, self.weight, None, add_input,
                                 name=self.name, sparse_unique_name=self._sparse_unique_name,
                                 act_type=self.act_type, act_alpha=self.act_alpha, act_beta=self.act_beta)
    finally:
        self.algo = algo
    return out if self.bias is None else out.replace_feature(out.features + self.bias)


_spconv_forward_orig = spconv.conv.SparseConvolution.forward
spconv.conv.SparseConvolution.forward = _spconv_forward_cpu_compat

# Enable TF32 precision if supported (for GPU >= Ampere and PyTorch >= 1.12)
if hasattr(torch.backends.cuda, "matmul") and hasattr(
    torch.backends.cuda.matmul, "allow_tf32"
):
    torch.backends.cuda.matmul.allow_tf32 = True

class DRSVPT(nn.Module, PyTorchModelHubMixin):
    "DRS VPT: localizes a camera image inside a point cloud map."

    def __init__(
        self,
        name: str,
        encoder_config: Dict,
        info_sharing_config: Dict,
        pred_head_config: Dict,
        geometric_input_config: Dict,
        fusion_norm_layer: Union[Type[nn.Module], Callable[..., nn.Module]] = partial(
            nn.LayerNorm, eps=1e-6
        ),
        pretrained_checkpoint_path: str = None,
        load_specific_pretrained_submodules: bool = False,
        specific_pretrained_submodules: list = None,
        torch_hub_force_reload: bool = False,
        skip_encoder_pretrain: bool = False,
        cloud_encoder_config: Dict = None,
        cloud_encoder_patch_size: int = 128,
        cloud_feature_dim: int = 1088,
        cloud_unpool_levels: int = 2,
        n_cloud_tokens: int = 5_000,
        cloud_head_blocks: int = 2,
        cloud_head_hidden_dim: int = 8,
        fuse_hidden_dim: int = 512,
        align_scales: Tuple[int, ...] = (1, 4, 16),
        max_align_points: Optional[int] = None,
        align_frustum_culling: bool = False,
        mask_threshold: float = 0.5,
    ):
        """
        Image encoder (DINOv2) and point cloud encoder (Sonata), fused by an alternating-attention transformer
        over image tokens, cloud tokens and a scale token. DPT dense heads predict per-pixel ray directions, depth,
        confidence and mask; a pose head the camera pose and an MLP head the metric scale. Coarse-to-fine feature heads
        and a PixLoc-style Levenberg-Marquardt aligner then refine the pose against the cloud.

        Args:
            name (str): Name of the model.
            encoder_config (Dict): Configuration for the encoder.
            info_sharing_config (Dict): Configuration for the multi-view attention transformer.
            pred_head_config (Dict): Configuration for the prediction heads.
            geometric_input_config (Dict): Only its scale_encoder_config is used, for the cloud scale encoder.
            fusion_norm_layer (Union[Type[nn.Module], Callable[..., nn.Module]]): Normalization layer used by the cloud adaptor. (default: partial(nn.LayerNorm, eps=1e-6))
            pretrained_checkpoint_path (str): Path to pretrained checkpoint. (default: None)
            load_specific_pretrained_submodules (bool): Whether to load specific pretrained submodules. (default: False)
            specific_pretrained_submodules (list): List of specific pretrained submodules to load. Must be provided when load_specific_pretrained_submodules is True. (default: None)
            torch_hub_force_reload (bool): Whether to force reload the encoder from torch hub. (default: False)
            skip_encoder_pretrain (bool): Skip loading pretrained weights for sub-encoders (DINOv2, Sonata) during __init__.
                Set True when loading a full model checkpoint immediately after, to avoid redundant disk I/O. (default: False)
                DINOv2 is then built from the vendored copy in drs_vpt._vendor.dinov2 (no GitHub download).
            cloud_encoder_config (Dict): Sonata (PointTransformerV3) constructor config. With skip_encoder_pretrain,
                providing it avoids downloading the facebook/sonata checkpoint just to read its config. (default: None)
            cloud_encoder_patch_size (int): Sonata attention patch size, used when its config comes from facebook/sonata.
                (default: 128)
            cloud_feature_dim (int): Channels of the Sonata features after un-pooling (512 + 384 + 192). (default: 1088)
            cloud_unpool_levels (int): Sonata pooling levels undone, concatenating features at each. (default: 2)
            n_cloud_tokens (int): Cloud tokens sampled for the transformer. (default: 5000)
            cloud_head_blocks (int): Residual blocks before and after the 1x1 conv in each cloud head. (default: 2)
            cloud_head_hidden_dim (int): Hidden width of those blocks. (default: 8)
            fuse_hidden_dim (int): Hidden width of the block that fuses token features back onto every cloud point.
                (default: 512)
            align_scales (Tuple[int, ...]): Downsampling factors of the feature maps for coarse-to-fine alignment,
                finest first; alignment runs coarsest first. (default: (1, 4, 16))
            max_align_points (Optional[int]): Cap on the cloud points the aligner uses per sample; None uses every
                point. (default: None)
            align_frustum_culling (bool): The aligner uses only the cloud points inside the camera frustum of the
                initial pose. (default: False)
            mask_threshold (float): Threshold on the predicted non-ambiguous mask. (default: 0.5)
        """
        super().__init__()

        # Initalize the attributes
        self.name = name
        self.encoder_config = encoder_config
        self.info_sharing_config = info_sharing_config
        self.pred_head_config = pred_head_config
        self.geometric_input_config = geometric_input_config
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        self.load_specific_pretrained_submodules = load_specific_pretrained_submodules
        self.specific_pretrained_submodules = specific_pretrained_submodules
        self.torch_hub_force_reload = torch_hub_force_reload
        self.class_init_args = {
            "name": self.name,
            "encoder_config": self.encoder_config,
            "info_sharing_config": self.info_sharing_config,
            "pred_head_config": self.pred_head_config,
            "geometric_input_config": self.geometric_input_config,
            "pretrained_checkpoint_path": self.pretrained_checkpoint_path,
            "load_specific_pretrained_submodules": self.load_specific_pretrained_submodules,
            "specific_pretrained_submodules": self.specific_pretrained_submodules,
            "torch_hub_force_reload": self.torch_hub_force_reload,
        }

        # Get relevant parameters from the configs
        self.info_sharing_type = info_sharing_config["model_type"]
        self.info_sharing_return_type = info_sharing_config["model_return_type"]
        self.pred_head_type = pred_head_config["type"]

        self._initialize_image_encoder(torch_hub_force_reload, skip_encoder_pretrain)
        self._initialize_cloud_encoder(skip_encoder_pretrain, cloud_encoder_config, cloud_encoder_patch_size,
                                       cloud_feature_dim)

        # Initialize the fusion norm layer
        self.fusion_norm_layer = fusion_norm_layer(self.encoder.enc_embed_dim)

        # Initialize the Scale Token
        # Used to scale the final scene predictions to metric scale
        # During inference extended to (B, C, T), where T is the number of tokens (i.e., 1)
        self.scale_token = nn.Parameter(torch.zeros(self.encoder.enc_embed_dim))
        torch.nn.init.trunc_normal_(self.scale_token, std=0.02)

        # Initialize the info sharing module (multi-view transformer)
        self._initialize_info_sharing(info_sharing_config)

        # Initialize the prediction heads
        self._initialize_prediction_heads(pred_head_config)

        # Initialize the final adaptors
        self._initialize_adaptors(pred_head_config)

        cloud_scale_encoder_config = self.geometric_input_config["scale_encoder_config"]
        cloud_scale_encoder_config["enc_embed_dim"] = self.encoder.enc_embed_dim
        self.input_cloud_scale_encoder = encoder_factory(**cloud_scale_encoder_config)
        self.input_cloud_scale_gate = nn.Parameter(torch.tensor(0.0))

        self.cloud_unpool_levels = cloud_unpool_levels
        self.n_cloud_tokens = n_cloud_tokens
        self.max_align_points = max_align_points
        self.align_frustum_culling = align_frustum_culling
        self.mask_threshold = mask_threshold
        self.align_scales = list(align_scales)
        self.n_align_scales = len(self.align_scales)

        self._initialize_image_align_heads(pred_head_config)
        self._initialize_cloud_heads(pred_head_config, cloud_head_blocks, cloud_head_hidden_dim, fuse_hidden_dim)

        # Load pretrained weights
        self._load_pretrained_weights()

    @classmethod
    def _load_as_safetensor(cls, model, model_file: str, map_location: str, strict: bool):
        # The file stores tensors shared under several names (e.g. the DPT scratch convs) once, so load non-strictly;
        # a missing key is fine when it shares storage with a loaded one.
        import safetensors.torch

        missing, unexpected = model.load_state_dict(safetensors.torch.load_file(model_file, device=map_location),
                                                    strict=False)
        model_state = model.state_dict()
        loaded_ptrs = {v.data_ptr() for k, v in model_state.items() if k not in set(missing)}
        missing = [k for k in missing if model_state[k].data_ptr() not in loaded_ptrs]
        if strict and (missing or unexpected):
            raise RuntimeError(f"Checkpoint does not match the model: missing {missing}, unexpected {unexpected}")
        model.eval()
        return model

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def _initialize_image_encoder(self, torch_hub_force_reload, skip_encoder_pretrain):
        """Initialize the DINOv2 image encoder."""
        if self.encoder_config["uses_torch_hub"]:
            self.encoder_config["torch_hub_force_reload"] = torch_hub_force_reload
        # Create a copy of the config before deleting the key to preserve it for serialization
        encoder_config_copy = self.encoder_config.copy()
        del encoder_config_copy["uses_torch_hub"]
        if skip_encoder_pretrain and self.encoder_config.get("uses_torch_hub"):
            # Weights come from the full checkpoint, so build the untrained architecture from the vendored
            # DINOv2 code instead of fetching facebookresearch/dinov2 from GitHub via torch.hub.
            import torch.hub as _torch_hub
            from drs_vpt._vendor.dinov2.hub import backbones as _dinov2_backbones
            _orig_hub_load = _torch_hub.load
            _torch_hub.load = lambda repo, name, *a, **kw: getattr(_dinov2_backbones, name)(pretrained=False)
            try:
                self.encoder = encoder_factory(**encoder_config_copy)
            finally:
                _torch_hub.load = _orig_hub_load
        else:
            self.encoder = encoder_factory(**encoder_config_copy)

    def _initialize_cloud_encoder(self, skip_encoder_pretrain, cloud_encoder_config, cloud_encoder_patch_size,
                                  cloud_feature_dim):
        """Initialize the Sonata cloud encoder and the projection of its features to the image token width."""
        sonata_config = dict(
            enc_patch_size=[cloud_encoder_patch_size for _ in range(5)],
            enable_flash=False,
        )
        # Use local cache if available to avoid a redundant HuggingFace network check
        _sonata_cache = os.path.expanduser("~/.cache/sonata/ckpt/sonata.pth")
        _sonata_name = _sonata_cache if os.path.isfile(_sonata_cache) else "sonata"
        if skip_encoder_pretrain:
            from drs_vpt._vendor.sonata.model import PointTransformerV3
            if cloud_encoder_config is None:
                cloud_encoder_config = sonata.load(_sonata_name, repo_id="facebook/sonata",
                                                   custom_config=sonata_config, ckpt_only=True)["config"]
            self.cloud_encoder = PointTransformerV3(**cloud_encoder_config)
        else:
            self.cloud_encoder = sonata.load(
                _sonata_name, repo_id="facebook/sonata", custom_config=sonata_config
            )
        self.cloud_encoder_embed = nn.Linear(in_features=cloud_feature_dim, out_features=self.encoder.model.embed_dim)

    def _initialize_info_sharing(self, info_sharing_config):
        """
        Initialize the information sharing module based on the configuration.

        This method sets up the custom positional encoding if specified and initializes
        the appropriate multi-view transformer based on the configuration type.

        Args:
            info_sharing_config (Dict): Configuration for the multi-view attention transformer.
                Should contain 'custom_positional_encoding', 'model_type', and 'model_return_type'.

        Returns:
            None

        Raises:
            ValueError: If invalid configuration options are provided.
        """
        if info_sharing_config["custom_positional_encoding"] is not None:
            raise ValueError("custom_positional_encoding is not supported by DRS VPT")
        self.custom_positional_encoding = None

        # Add dependecies to info_sharing_config
        info_sharing_config["module_args"]["input_embed_dim"] = (
            self.encoder.enc_embed_dim
        )
        info_sharing_config["module_args"]["custom_positional_encoding"] = (
            self.custom_positional_encoding
        )

        # Initialize the multi-view transformer: alternating attention returning two intermediate layers, which feed the
        # DPT heads together with the image encoder features
        if self.info_sharing_type != "alternating_attention" or self.info_sharing_return_type != "intermediate_features":
            raise ValueError("DRS VPT uses info_sharing type 'alternating_attention' with 'intermediate_features', got "
                             f"{self.info_sharing_type!r} / {self.info_sharing_return_type!r}")
        self.info_sharing = MultiViewAlternatingAttentionTransformerIFR(**info_sharing_config["module_args"])
        if len(self.info_sharing.indices) != 2:
            raise ValueError("DRS VPT's DPT heads take the encoder features plus 2 intermediate info sharing layers")

    def _initialize_prediction_heads(self, pred_head_config):
        """
        Initialize the prediction heads based on the prediction head configuration.

        This method configures and initializes the appropriate prediction heads based on the
        specified prediction head type (linear, DPT, or DPT+pose). It sets up the necessary
        dependencies and creates the required model components.

        Args:
            pred_head_config (Dict): Configuration for the prediction heads.

        Returns:
            None

        Raises:
            ValueError: If an invalid pred_head_type is provided.
        """
        if self.pred_head_type != "dpt+pose":
            raise ValueError(f"DRS VPT uses pred_head_type 'dpt+pose', got {self.pred_head_type!r}")
        # Add dependencies to prediction head config: the DPT heads take the encoder features and three info sharing
        # layers; the pose and scale heads take the info sharing features
        pred_head_config["feature_head"]["patch_size"] = self.encoder.patch_size
        pred_head_config["feature_head"]["input_feature_dims"] = [self.encoder.enc_embed_dim] + [self.info_sharing.dim] * 3
        pred_head_config["regressor_head"]["input_feature_dim"] = pred_head_config["feature_head"]["feature_dim"]
        pred_head_config["pose_head"]["patch_size"] = self.encoder.patch_size
        pred_head_config["pose_head"]["input_feature_dim"] = self.info_sharing.dim
        pred_head_config["scale_head"]["input_feature_dim"] = self.info_sharing.dim

        # Initialize the dense (DPT feature + regression), pose and scale heads
        self.dense_head = nn.Sequential(DPTFeature(**pred_head_config["feature_head"]),
                                        DPTRegressionProcessor(**pred_head_config["regressor_head"]))
        self.pose_head = PoseHead(**pred_head_config["pose_head"])
        self.scale_head = MLPHead(**pred_head_config["scale_head"])

    def _initialize_adaptors(self, pred_head_config):
        """
        Initialize the adaptors based on the prediction head configuration.

        The dense head predicts ray directions + depth along the ray with confidence and mask, the pose head the camera
        translation + quaternion, and the scale head the metric scale.

        Args:
            pred_head_config (Dict): Configuration for the prediction heads including adaptor type.

        Returns:
            None

        Raises:
            ValueError: If an invalid adaptor_type is provided.
        """
        self.cloud_adaptor = RayDirectionsPlusDepthWithConfidenceAndMaskAdaptor(**pred_head_config["cloud_adaptor"])

        if pred_head_config["adaptor_type"] != "raydirs+depth+pose+confidence+mask":
            raise ValueError("DRS VPT predicts 'raydirs+depth+pose+confidence+mask', got adaptor_type "
                             f"{pred_head_config['adaptor_type']!r}")
        self.dense_adaptor = RayDirectionsPlusDepthWithConfidenceAndMaskAdaptor(**pred_head_config["dpt_adaptor"])
        self.pose_adaptor = CamTranslationPlusQuatsAdaptor(**pred_head_config["pose_adaptor"])
        self.scale_adaptor = ScaleAdaptor(**pred_head_config["scale_adaptor"])

    def _initialize_image_align_heads(self, pred_head_config):
        """Initialize the aligner and the coarse-to-fine image feature heads it aligns, one per align scale."""
        self.aligner = LearnedOptimizer()
        self.align_image_adaptor = AlignFeatureWithConfidenceAdaptor(
            **pred_head_config["align_adaptor"]
        )

        self.align_image_heads = nn.ModuleList()

        for _ in range(self.n_align_scales):
            feature_head = DPTFeature(**pred_head_config["feature_head"])

            reg_cfg = copy.deepcopy(pred_head_config["regressor_head"])
            reg_cfg["output_dim"] = self.align_image_adaptor.required_channels
            regressor_head = DPTRegressionProcessor(**reg_cfg)

            self.align_image_heads.append(
                nn.Sequential(feature_head, regressor_head)
            )

    def _initialize_cloud_heads(self, pred_head_config, cloud_head_blocks, cloud_head_hidden_dim, fuse_hidden_dim):
        """Initialize the per-point cloud heads: the fusion of token features back onto every cloud point, the
        coarse-to-fine cloud feature heads for alignment, and the cloud ray/depth prediction head."""
        self.fuse_upsample_3d = ZeroInitResidualFuse(
            Cd=self.encoder.model.embed_dim,  # per-point cloud features
            Cs=self.info_sharing.dim,  # cloud token features after the transformer
            hidden=fuse_hidden_dim,
            use_s_in_delta=True
        )

        self.align_cloud_adaptor = AlignFeatureWithConfidenceAdaptor(
            **pred_head_config["align_adaptor"]
        )

        self.align_cloud_heads = nn.ModuleList([
            make_wrapped_conv1x1(
                in_channels=self.info_sharing.dim,
                out_channels=self.align_cloud_adaptor.required_channels,
                pre_blocks=cloud_head_blocks,
                post_blocks=cloud_head_blocks,
                pre_hidden=cloud_head_hidden_dim,
                post_hidden=cloud_head_hidden_dim,
                act="gelu",
            )
            for _ in range(self.n_align_scales)
        ])

        self.cloud_prediction_head = make_wrapped_conv1x1(
            in_channels=self.info_sharing.dim,
            out_channels=self.cloud_adaptor.required_channels,
            pre_blocks=cloud_head_blocks,
            post_blocks=cloud_head_blocks,
            pre_hidden=cloud_head_hidden_dim,
            post_hidden=cloud_head_hidden_dim,
            act="gelu",
        )

    def _load_pretrained_weights(self):
        """
        Load pretrained weights from a checkpoint file.

        If load_specific_pretrained_submodules is True, only loads weights for the specified submodules.
        Otherwise, loads all weights from the checkpoint.

        Returns:
            None
        """
        if self.pretrained_checkpoint_path is not None:
            if not self.load_specific_pretrained_submodules:
                print(f"Loading pretrained DRS VPT weights from {self.pretrained_checkpoint_path} ...")
                ckpt = torch.load(self.pretrained_checkpoint_path, weights_only=False)
                print(self.load_state_dict(ckpt["model"]))
            else:
                print(f"Loading pretrained DRS VPT weights from {self.pretrained_checkpoint_path} for specific submodules: "
                      f"{self.specific_pretrained_submodules} ...")
                assert self.specific_pretrained_submodules is not None, "Specific submodules to load cannot be None."
                ckpt = torch.load(self.pretrained_checkpoint_path, weights_only=False)
                filtered_ckpt = {}
                for ckpt_key, ckpt_value in ckpt["model"].items():
                    for submodule in self.specific_pretrained_submodules:
                        if ckpt_key.startswith(submodule):
                            filtered_ckpt[ckpt_key] = ckpt_value
                print(self.load_state_dict(filtered_ckpt, strict=False))

    def _encode_n_views(self, image_views, cloud=None, map_encoding=None):
        """
        Encode all the input views (batch of images) in a single forward pass.
        Assumes all the input views have the same image shape, batch size, and data normalization type.

        Args:
            image_views (List[dict]): One dict per image view, with its images and instance information.
            cloud (dict, optional): The collated cloud input, encoded here when map_encoding is None. With neither,
                no cloud tokens are used (camera-only).
            map_encoding (dict, optional): Output of encode_map for the cloud.

        Returns:
            List[torch.Tensor]: A list containing the encoded features for all N views.
        """
        num_views = len(image_views)
        data_norm_type = image_views[0]["data_norm_type"][0]
        imgs_list = [view["img"] for view in image_views]
        all_imgs_across_views = torch.cat(imgs_list, dim=0)
        encoder_input = ViTEncoderInput(
            image=all_imgs_across_views, data_norm_type=data_norm_type
        )
        encoder_output = self.encoder(encoder_input)
        all_camera_encoder_features_across_views = encoder_output.features.chunk(
            num_views, dim=0
        )

        if cloud is None and map_encoding is None:  # camera-only: no cloud tokens
            no_cloud = all_camera_encoder_features_across_views[0].new_zeros(
                all_camera_encoder_features_across_views[0].shape[0], 0, self.info_sharing.input_embed_dim)
            return all_camera_encoder_features_across_views, no_cloud, None, None, None, None
        if map_encoding is None:
            map_encoding = self.encode_map(cloud)
        return (all_camera_encoder_features_across_views, map_encoding["feats"], map_encoding["coords"],
                map_encoding["cloud_view"], map_encoding["inv_gather"], map_encoding["scene_scale_per_point"])

    def encode_map(self, single_cloud_view):
        """Encode the cloud (Sonata + token sampling). It does not depend on the image, so for a stream of
        images against one map it can be computed once and passed to forward(..., map_encoding=...).

        The cloud encoder runs on the device of its own parameters (it may be kept on the CPU, since spconv has
        no MPS kernels); the results are moved to the device of the rest of the model.
        """
        cloud_device = next(self.cloud_encoder.parameters()).device
        model_device = self.scale_token.device
        if cloud_device != model_device:
            single_cloud_view = {k: v.to(cloud_device) if torch.is_tensor(v) else v
                                 for k, v in single_cloud_view.items()}
        if self.training:
            cloud_view = checkpoint(lambda v: self.cloud_encoder(v), single_cloud_view, use_reentrant=True)
        else:
            cloud_view = self.cloud_encoder(single_cloud_view)
        for _ in range(self.cloud_unpool_levels):
            assert "pooling_parent" in cloud_view.keys()
            assert "pooling_inverse" in cloud_view.keys()
            parent = cloud_view.pop("pooling_parent")
            inverse = cloud_view.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, cloud_view.feat[inverse]], dim=-1)
            cloud_view = parent
        if cloud_device != model_device:
            for k, v in list(cloud_view.items()):
                if torch.is_tensor(v):
                    cloud_view[k] = v.to(model_device)
                elif isinstance(v, (list, tuple)) and v and all(torch.is_tensor(x) for x in v):
                    cloud_view[k] = type(v)(x.to(model_device) for x in v)

        cloud_features = self.cloud_encoder_embed(cloud_view.feat)
        cloud_view.feat = cloud_features

        # Handle re-sampling and batching
        batch = cloud_view["batch"].to(torch.long)  # (N,)
        scene_scale = single_cloud_view["scene_scale"].to(model_device)  # (B,)
        scene_scale_per_point = scene_scale[batch]

        # 1) Build sampling + inverse once
        sample_idx, inv_gather, batch_ids, sample_pos, orig_pos = sample_evenly_with_inverse(
            batch=cloud_view["batch"].long(),
            serialized_order=cloud_view.serialized_order[0].long(),
            K=self.n_cloud_tokens,
        )

        cloud_scale_feat = (
            self.input_cloud_scale_gate *
            self.input_cloud_scale_encoder(scene_scale.reshape(-1, 1)).features
        )
        # Gather sampled coords/features
        cloud_token_coords = cloud_view.coord[sample_idx]  # (B, n_cloud_tokens, 3)
        cloud_token_feats = cloud_view.feat[sample_idx] + cloud_scale_feat[:, None, :]  # (B, n_cloud_tokens, C)

        return {"feats": cloud_token_feats, "coords": cloud_token_coords, "cloud_view": cloud_view, "inv_gather": inv_gather,
                "scene_scale_per_point": scene_scale_per_point}


    def downstream_dense_head(
        self,
        dense_head_inputs: Union[torch.Tensor, List[torch.Tensor]],
        img_shape: Tuple[int, int],
    ):
        """
        Run the downstream dense prediction head
        """
        dense_head_outputs = self.dense_head(
            PredictionHeadLayeredInput(
                list_features=dense_head_inputs,
                target_output_shape=img_shape,
            )
        )
        dense_final_outputs = self.dense_adaptor(
            AdaptorInput(
                adaptor_feature=dense_head_outputs.decoded_channels,
                output_shape_hw=img_shape,
            )
        )

        align_c2f_output_shapes = [
            (int(img_shape[0] / scale), int(img_shape[1] / scale))
            for scale in reversed(self.align_scales)
        ]
        align_image_head_outputs = [image_head(
            PredictionHeadLayeredInput(
                list_features=dense_head_inputs, target_output_shape=target_shape
            )
        )
            for image_head, target_shape in zip(self.align_image_heads, align_c2f_output_shapes)]

        align_image_head_outputs = [
            self.align_image_adaptor(
                AdaptorInput(adaptor_feature=x.decoded_channels, output_shape_hw=y))
            for x, y in zip(align_image_head_outputs, align_c2f_output_shapes)
        ]

        return dense_final_outputs, align_image_head_outputs

    def downstream_head(
        self,
        dense_head_inputs: Union[torch.Tensor, List[torch.Tensor]],
        scale_head_inputs: torch.Tensor,
        img_shape: Tuple[int, int],
    ):
        """
        Run Prediction Heads & Post-Process Outputs
        """
        # Dense and pose prediction for all (batch_size * num_views) in one go
        dense_final_outputs, align_image_head_outputs = self.downstream_dense_head(dense_head_inputs, img_shape)
        pose_head_outputs = self.pose_head(PredictionHeadInput(last_feature=dense_head_inputs[-1]))
        pose_final_outputs = self.pose_adaptor(
            AdaptorInput(adaptor_feature=pose_head_outputs.decoded_channels, output_shape_hw=img_shape))

        # Scale prediction is lightweight, so we can run it in one go
        scale_head_output = self.scale_head(
            PredictionHeadTokenInput(last_feature=scale_head_inputs)
        )
        scale_final_output = self.scale_adaptor(
            AdaptorInput(
                adaptor_feature=scale_head_output.decoded_channels,
                output_shape_hw=img_shape,
            )
        )
        scale_final_output = scale_final_output.value.squeeze(-1)  # (B, 1, 1) -> (B, 1)


        return dense_final_outputs, pose_final_outputs, scale_final_output, align_image_head_outputs

    def forward(self, image_views, cloud=None, align_in_predicted_scale=False, map_encoding=None):
        """
        Forward pass:
        1. Encodes the N input images (DINOv2) and the cloud (Sonata), unless map_encoding is given.
        2. Shares information across image tokens, cloud tokens and a scale token (alternating-attention transformer).
        3. Runs the dense, pose and scale heads on the transformer features.
        4. Refines the pose by coarse-to-fine feature alignment of the image to the cloud.

        Assumption:
        - All the input views have the same image shape.

        Args:
            image_views (List[dict]): One dict per view, with:
                                    "img" (tensor): Image tensor of shape (B, C, H, W), normalized for the image encoder.
                                    "data_norm_type" (list): [model.encoder.data_norm_type]
            align_in_predicted_scale (bool): Align in the model's predicted scale instead of the cloud's metric
                scale. (default: False)
            cloud (dict, optional): The collated cloud input (Sonata fields and scene_scale). With neither cloud
                nor map_encoding the model runs camera-only: the per-view outputs only, and "cloud" is None.
            map_encoding (dict, optional): Cached output of encode_map, to localize several images in one map
                without re-encoding the cloud.

        Returns:
            dict: {"image_views": one dict of outputs per input view, "cloud": the cloud outputs (its predicted pose
                cloud_quats / cloud_trans, the aligned pose cloud_quats_opt / cloud_trans_opt per level, its points
                and confidences, and the alignment's cameras and reprojections)}.
        """
        camera_only = cloud is None and map_encoding is None

        # Get input shape of the images, number of views, and batch size per view
        batch_size_per_view, _, height, width = image_views[0]["img"].shape
        img_shape = (int(height), int(width))
        num_views = len(image_views)

        # Run the image encoder on all the input views
        (all_encoder_features_across_views,
         scene_cloud_encoder_features,
         scene_resampled_cloud_coords,
         cloud_view, inv_gather, scene_scale_per_point) = self._encode_n_views(image_views, cloud, map_encoding)

        # Expand the scale token to match the batch size
        input_scale_token = (
            self.scale_token.unsqueeze(0)
            .unsqueeze(-1)
            .repeat(batch_size_per_view, 1, 1)
        )  # (B, C, 1)

        # Combine all images and cloud features into view-centric representation
        # Output is a list containing the encoded features for all Nx2 views after information sharing.
        info_sharing_input = ImageCloudTransformerInput(
            camera_features=all_encoder_features_across_views,
            cloud_features=scene_cloud_encoder_features,
            additional_input_tokens=input_scale_token,
        )
        (
            final_camera_multi_view_feat,
            final_scene_cloud_feat,
            intermediate_image_cloud_info_sharing_multi_view_feat,
        ) = self.info_sharing(info_sharing_input)

        # Split cloud features from camera features
        final_info_sharing_multi_view_feat = MultiViewTransformerOutput(
            final_camera_multi_view_feat.features[:num_views],
            final_camera_multi_view_feat.additional_token_features)
        intermediate_info_sharing_multi_view_feat = [
            MultiViewTransformerOutput(x.features[:num_views], x.additional_token_features) for x in
            intermediate_image_cloud_info_sharing_multi_view_feat]

        # DPT inputs for all views: image encoder features, the two intermediate and the last info sharing layers
        dense_head_inputs = [
            torch.cat(all_encoder_features_across_views, dim=0),
            torch.cat(intermediate_info_sharing_multi_view_feat[0].features, dim=0),
            torch.cat(intermediate_info_sharing_multi_view_feat[1].features, dim=0),
            torch.cat(final_info_sharing_multi_view_feat.features, dim=0),
        ]

        with (torch.autocast("cuda", enabled=False)):
            scale_head_inputs = (
                final_info_sharing_multi_view_feat.additional_token_features
            )

            # Run the downstream heads
            dense_final_outputs, pose_final_outputs, scale_final_output, align_image_head_outputs = (
                self.downstream_head(
                    dense_head_inputs=dense_head_inputs,
                    scale_head_inputs=scale_head_inputs,
                    img_shape=img_shape,
                )
            )

            # Reshape output dense rep to (B * V, H, W, C)
            output_dense_rep = dense_final_outputs.value.permute(
                0, 2, 3, 1
            ).contiguous()
            # Get the predicted ray directions and depths along rays
            output_ray_directions, output_depth_along_ray = output_dense_rep.split(
                [3, 1], dim=-1
            )
            # Get the predicted camera translations and quaternions
            output_cam_translations, output_cam_quats = (
                pose_final_outputs.value.split([3, 4], dim=-1)
            )
            # Get the predicted pointmaps in world frame and camera frame
            output_pts3d = (
                convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
                    output_ray_directions,
                    output_depth_along_ray,
                    output_cam_translations,
                    output_cam_quats,
                )
            )
            output_pts3d_cam = output_ray_directions * output_depth_along_ray
            # Split the predicted quantities back to their respective views
            output_ray_directions_per_view = output_ray_directions.chunk(
                num_views, dim=0
            )
            output_depth_along_ray_per_view = output_depth_along_ray.chunk(
                num_views, dim=0
            )
            output_cam_translations_per_view = output_cam_translations.chunk(
                num_views, dim=0
            )
            output_cam_quats_per_view = output_cam_quats.chunk(num_views, dim=0)
            output_pts3d_per_view = output_pts3d.chunk(num_views, dim=0)
            output_pts3d_cam_per_view = output_pts3d_cam.chunk(num_views, dim=0)
            # Pack the output as a list of dictionaries
            res = []
            for i in range(num_views):
                res.append(
                    {
                        "pts3d": output_pts3d_per_view[i]
                                 * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                        "pts3d_cam": output_pts3d_cam_per_view[i]
                                     * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                        "ray_directions": output_ray_directions_per_view[i],
                        "depth_along_ray": output_depth_along_ray_per_view[i]
                                           * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                        "cam_trans": output_cam_translations_per_view[i]
                                     * scale_final_output,
                        "cam_quats": output_cam_quats_per_view[i],
                        "metric_scaling_factor": scale_final_output,
                    }
                )

            # Add the per-view confidences to the result
            output_confidences = dense_final_outputs.confidence
            # Reshape confidences to (B * V, H, W)
            output_confidences = (
                output_confidences.permute(0, 2, 3, 1).squeeze(-1).contiguous()
            )
            # Split the predicted confidences back to their respective views
            output_confidences_per_view = output_confidences.chunk(num_views, dim=0)
            # Add the confidences to the result
            for i in range(num_views):
                res[i]["conf"] = output_confidences_per_view[i]

            # Add the per-view masks (and mask logits) to the result
            # Get the output masks
            output_masks = dense_final_outputs.mask
            # Reshape masks to (B * V, H, W)
            output_masks = output_masks.permute(0, 2, 3, 1).squeeze(-1).contiguous()
            # Threshold the masks to get binary masks (0: ambiguous, 1: non-ambiguous)
            output_masks = output_masks > self.mask_threshold
            # Split the predicted masks back to their respective views
            output_masks_per_view = output_masks.chunk(num_views, dim=0)
            # Get the output mask logits (for loss)
            output_mask_logits = dense_final_outputs.logits
            # Reshape mask logits to (B * V, H, W)
            output_mask_logits = (
                output_mask_logits.permute(0, 2, 3, 1).squeeze(-1).contiguous()
            )
            # Split the predicted mask logits back to their respective views
            output_mask_logits_per_view = output_mask_logits.chunk(num_views, dim=0)
            # Add the masks and logits to the result
            for i in range(num_views):
                res[i]["non_ambiguous_mask"] = output_masks_per_view[i]
                res[i]["non_ambiguous_mask_logits"] = output_mask_logits_per_view[i]

            if camera_only:
                return {"image_views": res, "cloud": None}

            final_scene_cloud_feats_flat = final_scene_cloud_feat.reshape(-1, final_scene_cloud_feat.shape[
                -1])  # (B * n_cloud_tokens, C)

            upsampled_centroid_features = final_scene_cloud_feats_flat[inv_gather]

            densified_features = self.fuse_upsample_3d(upsampled_centroid_features,
                                                       cloud_view.feat)[None].permute(0, 2, 1).unsqueeze(-2)

            align_cloud_c2f_outputs = [self.align_cloud_heads[i](densified_features)
                                        for i in range(self.n_align_scales)]
            align_cloud_c2f_outputs = [self.align_cloud_adaptor(AdaptorInput(align_cloud_output,
                                                                               align_cloud_output.shape[-2:]))
                                        for align_cloud_output in align_cloud_c2f_outputs]
            align_cloud_features_c2f = list()
            align_cloud_confs_c2f = list()
            for cloud_output in align_cloud_c2f_outputs:
                align_cloud_features_c2f.append(dense_pack_simple(
                    cloud_output.value.squeeze().T, cloud_view.batch
                )[0])
                align_cloud_confs_c2f.append(dense_pack_simple(
                    cloud_output.confidence[0, :, 0, :].T, cloud_view.batch
                )[0])

            # Run cloud predictor
            cloud_head_output = self.cloud_prediction_head(
                densified_features
            )

            cloud_head_output = self.cloud_adaptor(AdaptorInput(cloud_head_output,
                                                                cloud_head_output.shape[-2:]))

            cloud_pose_head_outputs = self.pose_head(
                PredictionHeadInput(
                    last_feature=final_scene_cloud_feat.permute(0, 2, 1).unsqueeze(-2)
                )
            )

            cloud_pose_final_outputs = self.pose_adaptor(
                AdaptorInput(
                    adaptor_feature=cloud_pose_head_outputs.decoded_channels,
                    output_shape_hw=cloud_head_output.confidence.shape[-2:],
                )
            )
            cloud_translations, cloud_quats = (
                cloud_pose_final_outputs.value.split([3, 4], dim=-1)
            )

            cloud_dense_output = cloud_head_output.value.permute(0, 2, 3, 1).contiguous()
            cloud_ray_directions, cloud_ray_depths = (
                cloud_dense_output.split([3, 1], dim=-1)
            )
            cloud_pts3d = cloud_ray_directions * cloud_ray_depths
            cloud_ray_confs = cloud_head_output.confidence

            cloud_pts3d, valid_cloud_mask = dense_pack_simple(cloud_pts3d.squeeze(),
                                                              cloud_view.batch)
            densified_scene_coords, _ = dense_pack_simple(
                scene_scale_per_point[:, None] * cloud_view.coord, cloud_view.batch
            )
            B, _, _ = densified_scene_coords.shape
            model_predicted_pose = Pose.from_Rt(
                quaternion_to_rotation_matrix(cloud_quats.detach().clone()),
                cloud_translations.detach().clone()  # In predicted unit scale
            )
            real_scale2_pred_scale = torch.ones((B, 1)).to(densified_scene_coords).detach()
            for batch_index in range(B):
                ratios = (
                    torch.linalg.norm(
                        (model_predicted_pose.inv() * cloud_pts3d)[batch_index][valid_cloud_mask[batch_index]], axis=-1)
                    /
                    torch.linalg.norm(
                        (densified_scene_coords[batch_index][valid_cloud_mask[batch_index]]), axis=-1).clip(min=1e-8))

                real_scale2_pred_scale[batch_index] = torch.median(ratios).detach()

            align_scene_coords, align_valid_mask = densified_scene_coords, valid_cloud_mask

            # The encoder never sees intrinsics (geometric inputs are disabled), so the given K is only used by the
            # aligner below. Without one, use the pinhole K recovered from the predicted ray map.
            if "camera_intrinsics" in image_views[0]:
                K_cam0 = image_views[0]["camera_intrinsics"]
            else:
                K_cam0 = recover_pinhole_intrinsics_from_ray_directions(
                    output_ray_directions_per_view[0].float()).to(cloud_translations)
            cam_data = torch.cat([
                K_cam0.new_tensor([img_shape[1], img_shape[0]]).expand(K_cam0.shape[0], 2),
                K_cam0[:, [0, 1], [0, 1]],
                K_cam0[:, [0, 1], [2, 2]],
            ], dim=-1)
            camera_0 = Camera(cam_data)

            align_image_c2f_features_per_view = [
                x.value.chunk(num_views, dim=0) for x in align_image_head_outputs
            ]
            align_image_c2f_confidences_per_view = [
                x.confidence.chunk(num_views, dim=0) for x in align_image_head_outputs
            ]

            image_feature_shapes_wh = [x.value.shape[-2:][::-1] for x in align_image_head_outputs]

            optimized_poses = list()
            if align_in_predicted_scale:
                visible_points_for_alignment = align_scene_coords * real_scale2_pred_scale[:, None]
                T_init = model_predicted_pose
            else:
                visible_points_for_alignment = align_scene_coords.clone()
                T_init = Pose.from_Rt(model_predicted_pose.R.clone().detach(),
                                      model_predicted_pose.t.clone().detach() / real_scale2_pred_scale)

            # The cloud points the aligner uses: optionally only those inside the initial camera frustum, and at
            # most max_align_points per sample
            visible_in_camera = align_valid_mask
            if self.align_frustum_culling:
                _, in_frustum = camera_0.world2image(T_init * visible_points_for_alignment)
                visible_in_camera = visible_in_camera & in_frustum
            max_n_visible_points = visible_in_camera.sum(dim=1).max().item()
            if self.max_align_points is not None:
                max_n_visible_points = min(max_n_visible_points, self.max_align_points)
            idx = gather_valid_indices(visible_in_camera,
                                       max_n_visible_points)  # (B, max_n)
            # A sample with fewer points than max_n is padded by repeating its points; mask the repeats so that
            # each sample's alignment does not depend on the other samples in the batch.
            n_visible = visible_in_camera.sum(dim=1, keepdim=True)
            align_point_mask = torch.arange(idx.shape[1], device=idx.device)[None] < n_visible  # (B, max_n)
            visible_points_for_alignment = visible_points_for_alignment.gather(
                dim=1,
                index=idx.unsqueeze(-1).expand(-1, -1, visible_points_for_alignment.size(-1))
            )

            scene_coords_for_reprojection = align_scene_coords.gather(
                dim=1,
                index=idx.unsqueeze(-1).expand(-1, -1, align_scene_coords.size(-1))
            )

            cameras_at_level = list()
            reprojections_at_level = list()
            for i, _ in enumerate(self.align_scales):

                F_ref = align_cloud_features_c2f[i]
                if i > 0:
                    F_ref = F_ref + align_cloud_features_c2f[i - 1]
                F_ref = F_ref.gather(
                    dim=1,
                    index=idx.unsqueeze(-1).expand(-1, -1, F_ref.size(-1))
                )
                W_ref = align_cloud_confs_c2f[i]
                W_ref = W_ref.gather(
                    dim=1,
                    index=idx.unsqueeze(-1).expand(-1, -1, W_ref.size(-1))
                )

                F_query = align_image_c2f_features_per_view[i][0]
                if i > 0:
                    F_query = (F_query +
                               torch.nn.functional.interpolate(
                                   align_image_c2f_features_per_view[i - 1][0],
                                   size=image_feature_shapes_wh[i][::-1],
                                   mode="bilinear", align_corners=True))
                W_query = align_image_c2f_confidences_per_view[i][0]
                effective_scale = (
                    torch.tensor(image_feature_shapes_wh[i], device=F_query.device, dtype=F_query.dtype) /
                    torch.tensor(image_feature_shapes_wh[-1], device=F_query.device, dtype=F_query.dtype)
                )

                camera_at_level = camera_0.scale(effective_scale)
                cameras_at_level.append(camera_at_level)

                optimized_cloud_pose_at_level, _ = self.aligner.run_optimization(
                    p3D=visible_points_for_alignment,
                    F_ref=F_ref,
                    F_query=F_query,
                    T_init=T_init, camera=camera_at_level,
                    mask=align_point_mask,
                    W_ref_query=(W_ref, W_query))

                pr_projections = camera_at_level.world2image(
                    optimized_cloud_pose_at_level * visible_points_for_alignment)
                reprojections_at_level.append(pr_projections)
                T_init = optimized_cloud_pose_at_level.detach()
                optimized_poses.append(optimized_cloud_pose_at_level)

        cloud_out = {}
        cloud_out["resampled_cloud_coords"] = densified_scene_coords
        cloud_out["valid_cloud_mask"] = valid_cloud_mask
        cloud_out["cloud_pts3d"] = cloud_pts3d * scale_final_output.unsqueeze(-1)
        cloud_out["cloud_conf"] = cloud_ray_confs
        if align_in_predicted_scale:
            cloud_out["cloud_trans_opt"] = torch.stack(
                [x.t.detach() * scale_final_output  # Bare with me. This scale will be divided away in the loss fn
                 for x in optimized_poses], dim=1)
        else:
            cloud_out["cloud_trans_opt"] = torch.stack(
                [x.t.detach()  # Bare with me. This scale will be divided away in the loss fn
                 for x in optimized_poses], dim=1)
        cloud_out["cloud_quats_opt"] = torch.stack([rotation_matrix_to_quaternion(x.R) for x in optimized_poses],
                                                   dim=1).detach()
        cloud_out["cameras_at_level"] = cameras_at_level
        cloud_out["reprojections_at_level"] = reprojections_at_level
        cloud_out["scene_coords_for_reprojection"] = scene_coords_for_reprojection
        cloud_out["cloud_trans"] = cloud_translations * scale_final_output
        cloud_out["cloud_quats"] = cloud_quats
        res[0]["camera_intrinsics_used"] = K_cam0
        return {"image_views": res, "cloud": cloud_out}


