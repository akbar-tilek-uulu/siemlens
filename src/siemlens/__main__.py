"""Entry point for ``python -m siemlens``.

Kept separate from :mod:`siemlens.cli` so that importing the CLI module never
runs anything, and so the package can be executed without installing the
console script.
"""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
