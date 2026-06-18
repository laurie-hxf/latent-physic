"""Stateful GRU residual adapter.

The package owns the recurrent model and trainer. Shared Newton rollout,
feature, friction, and checkpoint utilities remain in
``pointnet_residual_adapter`` because they are adapter-agnostic infrastructure.
"""

from .model import StatefulGRUResidualPredictor
from .direct_state_model import StatefulGRUDirectStatePredictor

__all__ = ["StatefulGRUResidualPredictor", "StatefulGRUDirectStatePredictor"]
