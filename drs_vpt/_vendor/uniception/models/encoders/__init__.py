"""
Encoder Factory for UniCeption
"""

from drs_vpt._vendor.uniception.models.encoders.base import (
    EncoderGlobalRepInput,
    EncoderInput,
    UniCeptionEncoderBase,
    UniCeptionViTEncoderBase,
    ViTEncoderInput,
    ViTEncoderNonImageInput,
    ViTEncoderOutput,
)
from drs_vpt._vendor.uniception.models.encoders.dinov2 import DINOv2Encoder, DINOv2IntermediateFeatureReturner
from drs_vpt._vendor.uniception.models.encoders.global_rep_encoder import GlobalRepresentationEncoder

# Define encoder configurations
ENCODER_CONFIGS = {
    "dinov2": {
        "class": DINOv2Encoder,
        "intermediate_feature_returner_class": DINOv2IntermediateFeatureReturner,
        "supported_models": ["DINOv2", "DINOv2-Registers", "DINOv2-Depth-Anythingv2"],
    },
    "global_rep_encoder": {
        "class": GlobalRepresentationEncoder,
        "supported_models": ["Global-Representation-Encoder"],
    },
}


def encoder_factory(encoder_str: str, **kwargs) -> UniCeptionEncoderBase:
    """
    Encoder factory for UniCeption.
    Please use python3 -m uniception.models.encoders.list to see available encoders.

    Args:
        encoder_str (str): Name of the encoder to create.
        **kwargs: Additional keyword arguments to pass to the encoder constructor.

    Returns:
        UniCeptionEncoderBase: An instance of the specified encoder.
    """
    if encoder_str not in ENCODER_CONFIGS:
        raise ValueError(
            f"Unknown encoder: {encoder_str}. For valid encoder_str options, please use python3 -m uniception.models.encoders.list"
        )

    encoder_config = ENCODER_CONFIGS[encoder_str]
    encoder_class = encoder_config["class"]

    return encoder_class(**kwargs)


def feature_returner_encoder_factory(encoder_str: str, **kwargs) -> UniCeptionEncoderBase:
    """
    Factory for UniCeption Encoders with support for intermediate feature returning.
    Please use python3 -m uniception.models.encoders.list to see available encoders.

    Args:
        encoder_str (str): Name of the encoder to create.
        **kwargs: Additional keyword arguments to pass to the encoder constructor.

    Returns:
        UniCeptionEncoderBase: An instance of the specified encoder.
    """
    if encoder_str not in ENCODER_CONFIGS:
        raise ValueError(
            f"Unknown encoder: {encoder_str}. For valid encoder_str options, please use python3 -m uniception.models.encoders.list"
        )

    encoder_config = ENCODER_CONFIGS[encoder_str]
    encoder_class = encoder_config["intermediate_feature_returner_class"]

    return encoder_class(**kwargs)


def get_available_encoders() -> list:
    """
    Get a list of available encoders in UniCeption.

    Returns:
        list: A list of available encoder names.
    """
    return list(ENCODER_CONFIGS.keys())


def print_available_encoder_models():
    """
    Print the currently supported encoders in UniCeption.
    """
    print("Currently Supported Encoders in UniCeption:\nFormat -> encoder_str: supported_models")
    for encoder_name, config in ENCODER_CONFIGS.items():
        print(f"{encoder_name}: {', '.join(config['supported_models'])}")


__all__ = [
    "encoder_factory",
    "get_available_encoders",
    "print_available_encoder_models",
    "UniCeptionEncoderBase",
    "UniCeptionViTEncoderBase",
    "EncoderInput",
    "ViTEncoderInput",
    "ViTEncoderOutput",
]
