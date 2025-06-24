"""Entry-point for ``python -m apps.zero_dte.backtest``.

This tiny wrapper exists because we have *both*

    • apps/zero_dte/backtest.py          – user-facing CLI script
    • apps/zero_dte/backtest/            – internal simulator package

and Python’s module resolution gives priority to *packages* over modules.
Hence ``python -m apps.zero_dte.backtest`` resolves to the *package*
initialiser and fails unless we provide this file.

All actual work is delegated to the sibling top-level module
``apps.zero_dte.backtest`` to avoid code duplication.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

# Import the sibling *module* (not the current package).
_cli_mod: ModuleType = importlib.import_module("apps.zero_dte.backtest_cli")

if hasattr(_cli_mod, "main"):
    sys.exit(_cli_mod.main())
else:  # pragma: no cover – defensive guard
    raise SystemExit("apps.zero_dte.backtest has no 'main' callable")
