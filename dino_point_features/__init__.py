"""Utilities for assigning DINO image features to 3-D point clouds."""

from .dino_extractor import DinoFeatureExtractor, DinoFeatureMap
from .projection import (
    CameraObservation,
    PointDinoFeatures,
    camera_observation_from_frame,
    camera_observations_from_frames,
    DinoFeatureProjector,
)

__all__ = [
    "CameraObservation",
    "DinoFeatureExtractor",
    "DinoFeatureMap",
    "DinoFeatureProjector",
    "PointDinoFeatures",
    "camera_observation_from_frame",
    "camera_observations_from_frames",
]
