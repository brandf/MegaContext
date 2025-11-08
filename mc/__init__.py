"""
Lightweight MegaContext runtime package for the nanochat integration.

Phase 1 focuses on building tensor-first data structures and modules that can be
invoked from the existing nanochat training scripts via a simple `--mc` flag.
"""

from .config import MCConfig, MegaContextConfig, WorkingContextConfig
from .mega_context import MegaContextTree
from .working_context import WorkingContext
from .gistnet import GistNetBase, build_gistnet
from .lensnet import LensNetBase, build_lensnet
from .focus_allocator import FocusAllocatorBase, build_focus_allocator
from .runtime import MCController

__all__ = [
    "MCConfig",
    "MegaContextConfig",
    "WorkingContextConfig",
    "MegaContextTree",
    "WorkingContext",
    "GistNetBase",
    "LensNetBase",
    "FocusAllocatorBase",
    "build_gistnet",
    "build_lensnet",
    "build_focus_allocator",
    "MCController",
]
