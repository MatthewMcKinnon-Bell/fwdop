"""
fwdop package

Expose the GFwdOp class for convenient importing via:

    from fwdop import GFwdOp

This package is intentionally lightweight and simply re-exports the
implementation in `g_fwd_op.py`.
"""

from .g_fwd_op import GFwdOp

__all__ = ["GFwdOp"]
