"""Basic import test for the MKTONE bot."""

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT), str(ROOT.parent)])


def test_can_import_main() -> None:
    module = importlib.import_module("src.bot")
    assert hasattr(module, "main")
