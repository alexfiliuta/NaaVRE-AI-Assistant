"""Locate small demo labs in either the full workspace or the public checkout.

The public repository contains only synthetic lab fixtures. Real research data,
including the optional EV dataset, is supplied separately by the researcher.
"""
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
LABS_ROOT = (WORKSPACE / "Labs" if (WORKSPACE / "Labs").is_dir()
             else WORKSPACE / "Testing" / "fixtures" / "labs")
