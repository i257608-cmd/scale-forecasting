"""Download the Petrobras 3W dataset v2.0.0 into data/3w/.

WHY this script exists
----------------------
3W v2.0.0 is not shipped as a release asset or on Zenodo: the Parquet files are committed
directly to the GitHub repo under ``dataset/<label>/`` (~1.74 GB, plain git objects, no LFS).
So the cheapest reproducible download is a *sparse, shallow* git clone of only ``dataset/``
pinned to a known commit. Pinning matters: Petrobras keeps pushing to ``main``, and a thesis
result must be traceable to the exact bytes it was computed from.

Usage
-----
    python src/data/download_3w.py                 # pinned commit (default)
    python src/data/download_3w.py --commit main   # latest (not reproducible; log the hash!)

After download the script verifies ``dataset.ini`` says version 2.0.0 and prints instance
counts per label and source (real / simulated / hand-drawn), which must match the paper:
1,119 real / 1,089 simulated / 20 hand-drawn.
"""

from __future__ import annotations

import argparse
import configparser
import subprocess
from collections import Counter
from pathlib import Path

REPO_URL = "https://github.com/petrobras/3W.git"
# WHY pinned: HEAD of petrobras/3W on 2026-11-14 (dataset.ini -> 2.0.0). Change only on purpose.
DEFAULT_COMMIT = "826fec47d5ba59e93b1b91be572bb48641043a2e"
ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "data" / "3w"


def run(cmd: list[str], cwd: Path | None = None) -> str:
    """Run a git command and fail loudly - a silent partial download would poison every result."""
    return subprocess.run(cmd, cwd=cwd, check=True, text=True, capture_output=True).stdout.strip()


def download(commit: str, dest: Path = DEST) -> Path:
    """Sparse-clone only dataset/ at `commit`. Idempotent: re-running just re-checks out."""
    if not (dest / ".git").exists():
        dest.mkdir(parents=True, exist_ok=True)
        # WHY --filter=blob:none + sparse: skips toolkit code/notebooks history; only blobs
        # under dataset/ are fetched, so we pull ~1.7 GB instead of the whole repo history.
        run(["git", "init", "-q"], dest)
        run(["git", "remote", "add", "origin", REPO_URL], dest)
        run(["git", "sparse-checkout", "set", "dataset"], dest)
    # WHY --depth 1 on a single commit: we need one snapshot, not history.
    run(["git", "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", commit], dest)
    run(["git", "checkout", "-q", "FETCH_HEAD"], dest)
    print("3W commit:", run(["git", "rev-parse", "HEAD"], dest))
    return dest / "dataset"


def verify(ds: Path) -> Counter:
    """Check version + count instances by (label, source). Returns the counter for tests/logs."""
    ini = configparser.ConfigParser()
    ini.read(ds / "dataset.ini")
    version = ini["VERSION"]["DATASET"]
    # WHY assert: every number in the report assumes v2.0.0 (v1 had 1,984 instances, other vars).
    assert version == "2.0.0", f"expected 3W 2.0.0, got {version}"

    counts: Counter = Counter()
    for f in ds.glob("*/*.parquet"):
        # Source is encoded in the filename prefix: WELL-xxxxx_<ts>, SIMULATED_xxxxx, DRAWN_xxxxx.
        source = {"WELL": "real", "SIMULATED": "simulated", "DRAWN": "drawn"}[f.stem.split("-")[0].split("_")[0]]
        counts[(int(f.parent.name), source)] += 1

    totals = Counter()
    for (_, src), n in counts.items():
        totals[src] += n
    print(f"3W {version}: {dict(totals)}")
    for label in sorted({k[0] for k in counts}):
        print(f"  class {label}: " + ", ".join(f"{s}={counts[(label, s)]}" for s in ("real", "simulated", "drawn")))
    return counts


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--commit", default=DEFAULT_COMMIT, help="3W git commit/branch to fetch")
    p.add_argument("--dest", type=Path, default=DEST)
    a = p.parse_args()
    verify(download(a.commit, a.dest))
