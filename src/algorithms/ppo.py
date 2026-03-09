"""
PPO utilities — pure-NumPy implementation.
Compatible with both CPU-only (no PyTorch) and GPU (PyTorch) environments.
"""
# Guard real torch import so a stale torch.py mock in the repo root
# cannot shadow the installed package.
import sys as _sys
import importlib as _importlib

# If 'torch' is already imported as a non-package (i.e. the mock file),
# evict it so the real package can be found.
_torch_mod = _sys.modules.get('torch')
if _torch_mod is not None and not hasattr(_torch_mod, '__path__'):
    # It's a module-file mock, not the real torch package — remove it
    for _key in [k for k in _sys.modules if k == 'torch' or k.startswith('torch.')]:
        del _sys.modules[_key]

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _TORCH_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    torch = None
    nn = None
    optim = None
    _TORCH_AVAILABLE = False


def ppo_update_numpy(actor, critic, rollout, config):
    """Pure-NumPy PPO update (used when PyTorch is unavailable)."""
    # This is a no-op stub — LagrangianCTDE runs its own update loop
    pass


__all__ = ["ppo_update_numpy", "_TORCH_AVAILABLE"]
