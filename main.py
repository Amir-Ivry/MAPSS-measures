"""Compatibility wrapper for the historical ``python main.py`` command."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mapss.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
