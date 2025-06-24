"""Importable CLI shim for external entry-points.

Some tools (pytest entrypoints, Poetry scripts, `python -m`) resolve packages
before modules, so `apps.zero_dte.backtest` refers to the *package* rather than
`backtest.py`.  We expose an alias module with a `main()` that boots the real
CLI from the neighbouring *file* to keep backwards-compatibility.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_pkg_dir = Path(__file__).resolve().parent
_cli_path = _pkg_dir / "backtest.py"  # sibling file with CLI implementation

_spec = importlib.util.spec_from_file_location("apps.zero_dte._backtest_cli", _cli_path)
if _spec is None or _spec.loader is None:  # pragma: no cover – should never happen
    raise ImportError("Could not locate backtest.py for CLI bootstrap")

_cli_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _cli_mod  # register before exec to break cycles
_spec.loader.exec_module(_cli_mod)  # type: ignore[arg-type]

main = getattr(_cli_mod, "main", None)
if main is None:  # pragma: no cover
    raise ImportError("backtest.py does not define a 'main' callable")
