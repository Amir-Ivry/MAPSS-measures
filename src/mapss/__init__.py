"""Public API for MAPSS perceptual source-separation measures."""

from .api import MAPSSResult, evaluate, mapss

__version__ = "1.1.0"
__all__ = ["MAPSSResult", "evaluate", "mapss"]
