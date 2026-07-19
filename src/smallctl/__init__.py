from __future__ import annotations

import sys

if sys.version_info >= (3, 8):
    from importlib.metadata import version, PackageNotFoundError
else:  # pragma: no cover
    from importlib_metadata import version, PackageNotFoundError  # type: ignore

try:
    __version__ = version("smallctl")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
