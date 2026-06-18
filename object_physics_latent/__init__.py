"""Trajectory-conditioned object physics latent experiments."""

from .dataset import (
    ENCODER_FEATURE_SCHEMA,
    EncoderFeatureBatch,
    ObjectPhysicsDataset,
    ObjectSpec,
    ObjectTrainingSample,
    build_manifest,
    load_manifest,
    validate_manifest,
)
from .encoder import (
    ObjectLatentOutput,
    ObjectPhysicsEncoder,
    TrajectoryGRUEncoder,
    TrajectorySetEncoder,
    VisualPointSetEncoder,
    latent_regularization_losses,
    same_object_consistency_loss,
    symmetric_info_nce_loss,
)
from .friction_decoder import (
    LatentConditionedFrictionDecoder,
    build_point_conditioning_features,
)
from .model import (
    TrajectoryConditionedFrictionModel,
    TrajectoryConditionedFrictionOutput,
)

__all__ = [
    "ENCODER_FEATURE_SCHEMA",
    "EncoderFeatureBatch",
    "ObjectPhysicsDataset",
    "ObjectLatentOutput",
    "ObjectPhysicsEncoder",
    "ObjectSpec",
    "ObjectTrainingSample",
    "TrajectoryConditionedFrictionModel",
    "TrajectoryConditionedFrictionOutput",
    "TrajectoryGRUEncoder",
    "TrajectorySetEncoder",
    "VisualPointSetEncoder",
    "LatentConditionedFrictionDecoder",
    "build_point_conditioning_features",
    "build_manifest",
    "latent_regularization_losses",
    "load_manifest",
    "same_object_consistency_loss",
    "symmetric_info_nce_loss",
    "validate_manifest",
]
