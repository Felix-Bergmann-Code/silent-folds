"""Module entry point for ``python -m warpaudit``."""

from .cli import main

if __name__ == "__main__":  # pragma: no cover - exercised through subprocess tests
    raise SystemExit(main())
