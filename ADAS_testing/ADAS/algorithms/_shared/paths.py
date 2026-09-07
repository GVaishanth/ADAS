"""
paths.py
========
Portable path helpers shared by the eval scripts and the comparison table.

The comparison JSON stores a ``checkpoint`` path. If we store an absolute path
it is tied to whatever machine did the training (e.g. ``/home/user/...`` on a
Linux box or ``C:\\Users\\...`` on Windows) and is meaningless elsewhere. This
module rewrites such paths to be *relative to PROJECT_ROOT* so the same JSON is
useful on any machine, and can resolve a portable path back to a real file.
"""

from __future__ import annotations

from pathlib import Path

# The ADAS/ project root (parent of this package's parent directory).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Sensible roots to strip when storing a path portably.
_PORTABLE_ROOTS = (PROJECT_ROOT,)


def portable(path: str | Path) -> str:
    """Return ``path`` rewritten relative to PROJECT_ROOT when possible.

    ``/home/user/ADAS_testing/ADAS/models/vit/best.pt`` -> ``models/vit/best.pt``
    ``C:\\Users\\RC\\Downloads\\ADAS_testing_2\\ADAS\\models\\vit\\best.pt`` -> ``models/vit/best.pt``
    Anything outside the repo is returned unchanged.
    """
    p = Path(str(path))
    try:
        rel = p.resolve().relative_to(PROJECT_ROOT)
        return rel.as_posix()
    except (ValueError, OSError):
        return str(path)


def resolve(path: str | Path) -> Path:
    """Turn a stored (possibly portable) path back into an absolute Path.

    ``models/vit/best.pt`` -> ``<PROJECT_ROOT>/models/vit/best.pt``.
    If the path was absolute to begin with it is kept as-is.
    """
    p = Path(str(path))
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p
