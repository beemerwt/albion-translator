#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.argos_backend import REQUIRED_PAIRS, install_required_packages


def main() -> int:
    try:
        summary = install_required_packages(REQUIRED_PAIRS)
    except Exception as error:
        print(f"fatal: failed to install Argos packages: {error}", file=sys.stderr)
        return 1

    print("Argos package installation summary")
    print(f"  installed:   {summary.installed or ['<none>']}")
    print(f"  skipped:     {summary.skipped or ['<none>']}")
    print(f"  unavailable: {summary.unavailable or ['<none>']}")
    print(f"  failed:      {summary.failed or ['<none>']}")

    successful = len(summary.installed) + len(summary.skipped)
    return 0 if successful > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
