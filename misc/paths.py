"""Locate sources when saving experiment provenance after the directory move."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source_path(name):
    """The notebook helper stays at the root; legacy scripts live in misc."""
    return ROOT / name if name == 'utils.py' else ROOT / 'misc' / name
