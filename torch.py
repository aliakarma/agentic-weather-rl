"""
Minimal torch mock — provides just enough API surface for the training notebook.
Used when PyTorch is not installed (CPU-only numpy backend).
"""


class _CudaModule:
    def is_available(self):
        return False


class _nn:
    class Module:
        """Base class stub — not used at runtime, only satisfies isinstance checks."""
        def parameters(self):
            return iter([])

        def numel(self):
            return 0


cuda = _CudaModule()
nn = _nn()


def no_grad():
    """Context manager stub."""
    class _Ctx:
        def __enter__(self): return self
        def __exit__(self, *a): pass
    return _Ctx()
