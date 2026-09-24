"""Download the SKIPP'D benchmark (sky images + PV power) from Hugging Face.

Run from the project folder:
    python -m src.skippd          # test file + all 5 training files (~2.3 GB)
    python -m src.skippd --small  # test file + 1 training file (~540 MB), for quick experiments

Dataset: solarbench/SKIPPD on Hugging Face (Stanford), license CC BY 4.0.
64x64 sky images and PV power, 1-minute steps, March 2017 - October 2019.
"""
import shutil
import ssl
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "raw" / "skippd"
BASE = "https://huggingface.co/datasets/solarbench/SKIPPD/resolve/main/data/"
FILES = ["test-00000-of-00001.parquet"] + [f"train-0000{i}-of-00005.parquet" for i in range(5)]


def _context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _is_valid(path) -> bool:
    """A complete parquet file can be opened; a cut-off one can't."""
    try:
        import pyarrow.parquet as pq
        pq.ParquetFile(path)
        return True
    except Exception:
        return False


def download(files):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in files:
        target = OUT_DIR / name
        if target.exists() and _is_valid(target):
            print(f"already have {name}")
            continue
        print(f"downloading {name} ...", flush=True)
        tmp = target.with_suffix(".part")
        with urllib.request.urlopen(BASE + name, context=_context(), timeout=300) as r, open(tmp, "wb") as f:
            expected = int(r.headers.get("Content-Length", 0))
            shutil.copyfileobj(r, f, length=1 << 20)
        if expected and tmp.stat().st_size != expected:
            tmp.unlink()
            raise RuntimeError(f"{name} was cut off during download. Please run the command again.")
        tmp.rename(target)
        print(f"  saved {target.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    download(FILES[:2] if "--small" in sys.argv else FILES)
    print("Done:", OUT_DIR)
