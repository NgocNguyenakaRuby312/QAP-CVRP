"""
encoder/__init__.py
===================
Re-exports for the encoder package.
"""

from .feature_constructor  import FeatureBuilder
from .qap_encoder          import QAPEncoder, FullEncoder
from .amplitude_projection import AmplitudeProjection
from .rotation_mlp         import RotationMLP
from .rotation             import PerNodeRotation, apply_rotation, rotation_matrix_2d

# Backward-compat aliases
FeatureConstructor = FeatureBuilder
CVRPEncoder        = FullEncoder

__all__ = ["FullEncoder", "FeatureConstructor", "QAPEncoder",
           "AmplitudeProjection", "RotationMLP", "PerNodeRotation"]
