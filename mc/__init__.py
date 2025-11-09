"""
Lightweight MegaContext runtime package for the nanochat integration.

Phase 1 focuses on building tensor-first data structures and modules that can be
invoked from the existing nanochat training scripts via a simple `--mc` flag.
"""

from .config import MCConfig, MegaContextConfig, WorkingContextConfig
from .mega_context import MegaContextTree, build_mega_context
from .working_context import WorkingContext
from .gistnet import GistNetBase, build_gistnet
from .lensnet import build_lensnet
from .focus_allocator import FocusAllocatorBase, FocusAllocatorConfig, build_focus_allocator
from .runtime import MCController

__all__ = [
    "MCConfig",
    "MegaContextConfig",
    "WorkingContextConfig",
    "MegaContextTree",
    "WorkingContext",
    "GistNetBase",
    "FocusAllocatorBase",
    "FocusAllocatorConfig",
    "build_gistnet",
    "build_lensnet",
    "build_focus_allocator",
    "build_mega_context",
    "MCController",
]
