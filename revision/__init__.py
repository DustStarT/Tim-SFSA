"""Revised event-anchored Tim-SFSA experiment pipeline."""

import os


# CuBLAS requires this variable to be present before PyTorch creates a CUDA
# context when deterministic algorithms are enabled.  Respect an explicit
# user choice while providing the larger deterministic workspace by default.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from .config import FEATURES_24, HORIZONS_HOURS, LEGACY_18_FEATURES

__all__ = ["FEATURES_24", "LEGACY_18_FEATURES", "HORIZONS_HOURS"]
