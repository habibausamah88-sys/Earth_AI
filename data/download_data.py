# ─────────────────────────────────────────────────────────────────────────────
# EarthLens AI — Dataset download & verification
# Owner: Member 1 — Data Engineer
# ─────────────────────────────────────────────────────────────────────────────
"""Download the EuroSAT RGB dataset, extract it and verify the folder structure.

The dataset (~89 MB zip, 27,000 Sentinel-2 RGB tiles, 64x64 px, 10 classes) is
NOT stored in git. Every teammate runs this once; the extracted folder lands in
``data/EuroSAT/<ClassName>/*.jpg`` which all training code expects.

Usage (from the repository root):
    python data/download_data.py                    # download + extract + verify
    python data/download_data.py --verify-only      # only check an existing copy
    python data/download_data.py --data-dir data/EuroSAT
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

# Allow "python data/download_data.py" (no package context) to still import configs.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs import config as cfg  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) EarthLensAI/1.0"}

# Expected per-class counts in the full EuroSAT RGB release (Helber et al., 2019).
EXPECTED_COUNTS = {
    "AnnualCrop": 3000,
    "Forest": 3000,
    "HerbaceousVegetation": 3000,
    "Highway": 2500,
    "Industrial": 2500,
    "Pasture": 2000,
    "PermanentCrop": 2500,
    "Residential": 3000,
    "River": 2500,
    "SeaLake": 2500,
}


def _progress(block: int, bs: int, total: int) -> None:
    """Simple download progress reporter."""
    done = block * bs
    if total > 0 and block % 2000 == 0:
        print(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)


def download_zip(dest: Path, url: str) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url}")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f, length=1024 * 1024)
    size_mb = dest.stat().st_size / 1e6
    print(f"\n[download] saved {dest} ({size_mb:.1f} MB)")
    return dest


def extract_zip(zip_path: Path, dest_dir: Path) -> Path:
    import tempfile
    dest_dir = dest_dir.resolve()
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest_dir.parent) as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(zip_path) as z:
            for item in z.infolist():
                if not (root / item.filename).resolve().is_relative_to(root):
                    raise ValueError('Unsafe archive path')
            z.extractall(root)
        candidates = [p for p in [root, *root.rglob('*')] if p.is_dir() and all((p / c).is_dir() for c in cfg.CLASS_NAMES)]
        if len(candidates) != 1:
            raise ValueError('Could not locate unique EuroSAT RGB class directory')
        dest_dir.mkdir(exist_ok=True)
        for c in cfg.CLASS_NAMES:
            shutil.copytree(candidates[0] / c, dest_dir / c, dirs_exist_ok=True)
    return dest_dir


def verify_dataset(data_dir: Path) -> bool:
    """Check the 10 class folders exist and warn when counts differ from the paper."""
    if not data_dir.is_dir():
        print(f"[verify] {data_dir} does not exist yet.")
        return False
    found = sorted(p for p in data_dir.iterdir() if p.is_dir())
    print(f"[verify] {len(found)} class folders found under {data_dir}")
    total = 0
    ok = [d.name for d in found] == cfg.CLASS_NAMES
    for d in found:
        n = sum(1 for _ in d.glob("*.jpg")) + sum(1 for _ in d.glob("*.tif"))
        total += n
        ok = ok and n > 0
        expected = EXPECTED_COUNTS.get(d.name)
        flag = "" if expected is None or n == expected else f"  (paper: {expected})"
        print(f"  - {d.name:<22} {n:>6}{flag}")
    print(f"[verify] total images: {total} (full dataset: 27,000)")
    print("[verify] citation:", cfg.EUROSAT_CITATION)
    return ok and total >= 5000


def main() -> None:
    ap = argparse.ArgumentParser(description="Download and verify EuroSAT RGB.")
    ap.add_argument("--data-dir", default=str(cfg.DATA_DIR), help="final dataset folder")
    ap.add_argument("--zip-dir", default="data/_zip", help="where to keep the zip")
    ap.add_argument("--verify-only", action="store_true", help="do not download")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if args.verify_only:
        verify_dataset(data_dir)
        return

    if verify_dataset(data_dir):
        print("[download] dataset already present — nothing to do.")
        return

    zip_path = Path(args.zip_dir) / "EuroSAT.zip"
    if not zip_path.exists() or not zipfile.is_zipfile(zip_path):
        try:
            download_zip(zip_path, cfg.EUROSAT_ZIP_URL)
        except Exception as e:  # noqa: BLE001
            print(f"[download] primary source failed ({e}); trying fallback ...")
            download_zip(zip_path, cfg.EUROSAT_ZIP_URL_FALLBACK)
    extract_dir = extract_zip(zip_path, data_dir)
    if extract_dir != data_dir:
        data_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(extract_dir), str(data_dir))
    if not verify_dataset(data_dir):
        raise ValueError("Dataset verification failed")
    print("\n[done] You can now run:  python -m src.data --audit --data-dir", data_dir)


if __name__ == "__main__":
    main()
