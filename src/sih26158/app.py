"""Backward-compatible ASGI entrypoint.

The implementation lives in :mod:`sih26158.api.app`; this module preserves the
existing ``sih26158.app:app`` launch contract for scripts and operators.
"""

from .api.app import app, create_app

__all__ = ["app", "create_app"]
