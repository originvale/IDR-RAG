"""Reference implementation of Information-Deficit Routing (IDR)."""

from .allocation import (
    action_curve,
    dynamic_order,
    random_eligible_order,
    static_path_order,
)
from .features import FEATURE_NAMES, features

__all__ = [
    "FEATURE_NAMES",
    "action_curve",
    "dynamic_order",
    "features",
    "random_eligible_order",
    "static_path_order",
]

__version__ = "2.0.0"
