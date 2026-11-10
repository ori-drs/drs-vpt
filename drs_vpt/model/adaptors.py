# Copyright (C) 2024-present Oxford Robotics Institute.
# Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.
from drs_vpt._vendor.uniception.models.prediction_heads import UniCeptionAdaptorBase, AdaptorInput, RegressionAdaptorOutput
from drs_vpt._vendor.uniception.models.prediction_heads.adaptors import ValueWithConfidenceAdaptor, ConfidenceAdaptor


class AlignFeatureAdaptor(UniCeptionAdaptorBase):
    def __init__(self, name: str, feature_dimension:int):
        super().__init__(name, required_channels=feature_dimension)

    def forward(self, adaptor_input: AdaptorInput):
        """
        Forward pass for the PointMapAdaptor.

        Args:
            adaptor_input (AdaptorInput): Input to the adaptor. (B x C x H x W)
        Returns:
            AdaptorOutput: Output of the adaptor.
        """
        features = adaptor_input.adaptor_feature

        return RegressionAdaptorOutput(value=features)


class AlignFeatureWithConfidenceAdaptor(ValueWithConfidenceAdaptor):
    def __init__(
        self,
        name: str,
        feature_dimension: int,
        confidence_type: str,
        confidence_vmin: float,
        confidence_vmax: float,
        *args,
        **kwargs,
    ):
        """
        Adaptor for the PointMap with Confidence head in UniCeption.
        """
        align_feature_adaptor = AlignFeatureAdaptor(name=f"{name}", feature_dimension=feature_dimension)

        confidence_adaptor = ConfidenceAdaptor(
            name=f"{name}_confidence", confidence_type=confidence_type, vmin=confidence_vmin, vmax=confidence_vmax
        )

        super().__init__(name, value_adaptor=align_feature_adaptor,
                         confidence_adaptor=confidence_adaptor, *args, **kwargs)
