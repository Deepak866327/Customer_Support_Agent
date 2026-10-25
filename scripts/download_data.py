"""Fetch twcs.csv (516 MB).

    python scripts/download_data.py

Primary source is the Kaggle dataset `thoughtvector/customer-support-on-twitter`.
Kaggle requires credentials, so this defaults to a public mirror of the identical file
and falls back to the Kaggle CLI if you have it configured.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "raw" / "twcs.csv"
MIRROR = "https://huggingface.co/datasets/SunidhiSriram/twcs/resolve/main/twcs.csv"
EXPECTED_BYTES = 516_508_641


def _progress(block: int, block_size: int, total: int) -> None:
    done = block * block_size
    pct = min(100.0, 100.0 * done / total) if total > 0 else 0.0
    print(f"\r  {done / 1e6:7.1f} MB / {total / 1e6:.1f} MB  ({pct:5.1f}%)", end="")


def main() -> None:
    if DEST.exists() and DEST.stat().st_size == EXPECTED_BYTES:
        print(f"already present: {DEST}")
        return
    DEST.parent.mkdir(parents=True, exist_ok=True)

    if shutil.which("kaggle"):
        print("trying Kaggle CLI ...")
        rc = subprocess.call(
            [
                "kaggle", "datasets", "download",
                "-d", "thoughtvector/customer-support-on-twitter",
                "-f", "twcs/twcs.csv", "--unzip",
                "-p", str(DEST.parent),
            ]
        )
        if rc == 0 and DEST.exists():
            print(f"downloaded via Kaggle -> {DEST}")
            return
        print("Kaggle CLI unavailable or unauthenticated; using mirror.")

    print(f"downloading mirror -> {DEST}")
    urllib.request.urlretrieve(MIRROR, DEST, _progress)
    print()

    size = DEST.stat().st_size
    if size != EXPECTED_BYTES:
        sys.exit(f"size mismatch: got {size}, expected {EXPECTED_BYTES}")
    print(f"ok ({size:,} bytes)")


if __name__ == "__main__":
    main()
