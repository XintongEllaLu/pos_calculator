"""
=================================================================
Phase Transition Probability (PoS) Calculator
FILE 0: Setup + AACT Data Download
=================================================================

WHAT THIS FILE DOES:
  1. Installs all required Python packages
  2. Downloads exactly the AACT files you need (not the full 2GB dump)
  3. Saves them to data/raw/

WHY AACT:
  - Clean, structured, no scraping needed
  - 480,000+ trials with consistent fields
  - Label (success/failure) is already in the data
  - Updated daily by CTTI (non-profit backed by FDA & industry)

HOW TO RUN:
  python 00_setup.py

TAKES ABOUT: 3-5 minutes (downloads ~150MB of files)
=================================================================
"""

import subprocess, sys, os, zipfile, time
from datetime import datetime

# ── PACKAGES ─────────────────────────────────────────────────
PACKAGES = [
    "pandas",
    "numpy",
    "scipy",
    "statsmodels",
    "scikit-learn",
    "matplotlib",
    "requests",
]

def install_packages():
    print("Installing required packages...")
    for pkg in PACKAGES:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg, "-q"],
            check=True
        )
        print(f"  ✓ {pkg}")
    print()

# ── AACT FLAT FILE DOWNLOAD ───────────────────────────────────
# AACT publishes monthly snapshots at this URL pattern.
# We download only the 4 tables we actually need:
#   studies.txt         → phase, status, enrollment, dates, sponsor
#   interventions.txt   → drug type (drug / biologic / device)
#   browse_conditions.txt → MeSH-coded therapeutic area
#   design_groups.txt   → number of arms (RCT vs single-arm)
#
# Full archive index: https://aact.ctti-clinicaltrials.org/archive/pipe_files
# Monthly archives stay up permanently — we use a recent month.

AACT_BASE = "https://aact.ctti-clinicaltrials.org/static/static_db_copies/monthly"

# Use a stable monthly snapshot (update this date if link breaks)
SNAPSHOT_DATE = "20250201"  # YYYYMMDD — change if needed

RAW_DIR = "data/raw"

# The full ZIP contains all ~46 tables; we unzip only what we need
NEEDED_FILES = [
    "studies.txt",
    "interventions.txt",
    "browse_conditions.txt",
    "design_groups.txt",
    "eligibilities.txt",       # has gender, min/max age
    "sponsors.txt",            # sponsor type (industry vs NIH)
]


def _download_with_resume(url, dest_path, max_retries=5):
    """Download a file with resume support and retry on incomplete transfer."""
    import requests

    for attempt in range(1, max_retries + 1):
        existing_bytes = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
        headers = {"Range": f"bytes={existing_bytes}-"} if existing_bytes else {}

        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                if r.status_code == 416:
                    # Server says range not satisfiable — we already have everything
                    return True
                r.raise_for_status()

                total = int(r.headers.get("Content-Length", 0)) + existing_bytes
                downloaded = existing_bytes

                mode = "ab" if existing_bytes else "wb"
                with open(dest_path, mode) as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded / total * 100
                            print(f"\r  {pct:5.1f}%  {downloaded/1e6:.1f} MB", end="", flush=True)

            # Verify we got the full file
            actual = os.path.getsize(dest_path)
            if total and actual < total:
                print(f"\n  Incomplete ({actual}/{total} bytes), retrying ({attempt}/{max_retries})...")
                continue

            print(f"\n  ✓ Downloaded → {dest_path}")
            return True

        except Exception as e:
            print(f"\n  ✗ Attempt {attempt}/{max_retries} failed: {e}")
            if attempt == max_retries:
                return False
            print("  Retrying in 3 seconds...")
            time.sleep(3)

    return False


def download_aact():
    os.makedirs(RAW_DIR, exist_ok=True)

    # Check if files already exist
    existing = [f for f in NEEDED_FILES
                if os.path.exists(f"{RAW_DIR}/{f}")]
    if len(existing) == len(NEEDED_FILES):
        print("✓ AACT files already downloaded. Skipping.")
        return

    zip_name = f"{SNAPSHOT_DATE}_pipe-delimited-export.zip"
    zip_url  = f"{AACT_BASE}/{zip_name}"
    zip_path = f"{RAW_DIR}/{zip_name}"

    print(f"Downloading AACT snapshot: {zip_name}")
    print(f"URL: {zip_url}")
    print("(This may take 2–4 minutes on a normal connection)\n")

    ok = _download_with_resume(zip_url, zip_path)

    if not ok:
        print()
        print("  ── MANUAL DOWNLOAD INSTRUCTIONS ─────────────────────────")
        print("  1. Go to: https://aact.ctti-clinicaltrials.org/downloads")
        print("  2. Click the most recent monthly pipe-delimited ZIP")
        print("  3. Save it to: data/raw/")
        print("  4. Run this script again — it will extract automatically")
        print("  ──────────────────────────────────────────────────────────")
        return

    # Validate the ZIP before extracting
    import zipfile as _zf
    try:
        with _zf.ZipFile(zip_path, 'r') as zf:
            bad = zf.testzip()
        if bad:
            print(f"  ✗ ZIP is corrupt (first bad file: {bad}). Deleting and re-running may help.")
            os.remove(zip_path)
            return
    except Exception as e:
        print(f"  ✗ ZIP validation failed: {e}. Deleting corrupt file.")
        os.remove(zip_path)
        return

    # Extract only the files we need
    print(f"\nExtracting needed tables...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        all_names = zf.namelist()

        # Show actual ZIP contents to help debug name mismatches
        txt_files = [n for n in all_names if n.endswith(".txt")]
        print(f"  ZIP contains {len(all_names)} entries, {len(txt_files)} .txt files")
        if txt_files:
            print(f"  Sample entries: {txt_files[:8]}")

        for needed in NEEDED_FILES:
            # Match on filename stem (case-insensitive, ignore directory prefix)
            matches = [n for n in all_names
                       if os.path.basename(n).lower() == needed.lower()]
            if matches:
                zf.extract(matches[0], RAW_DIR)
                # Rename to flat name if nested
                src = f"{RAW_DIR}/{matches[0]}"
                dst = f"{RAW_DIR}/{needed}"
                if src != dst:
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    os.replace(src, dst)
                print(f"  ✓ {needed}")
            else:
                print(f"  ✗ Not found in ZIP: {needed}")

    # Only remove ZIP if all files extracted successfully
    extracted = [f for f in NEEDED_FILES if os.path.exists(f"{RAW_DIR}/{f}")]
    if len(extracted) == len(NEEDED_FILES):
        os.remove(zip_path)
        print(f"\n  ZIP removed. Raw files saved in: {RAW_DIR}/")
    else:
        missing = [f for f in NEEDED_FILES if f not in [os.path.basename(e) for e in extracted]]
        print(f"\n  ⚠ ZIP kept at {zip_path} — {len(missing)} files not extracted.")
        print(f"  Missing: {missing}")
        print(f"  Check the 'Sample entries' above for the actual filenames in the ZIP.")


def verify_files():
    print("\nVerifying files...")
    import pandas as pd
    ok = True
    for fname in NEEDED_FILES:
        path = f"{RAW_DIR}/{fname}"
        if not os.path.exists(path):
            print(f"  ✗ Missing: {path}")
            ok = False
            continue
        # Peek at row count
        df = pd.read_csv(path, sep="|", nrows=5, low_memory=False)
        size_mb = os.path.getsize(path) / 1_000_000
        print(f"  ✓ {fname:<35} {size_mb:6.1f} MB  cols={len(df.columns)}")
    return ok


if __name__ == "__main__":
    print("=" * 60)
    print("  PoS Calculator — Step 0: Setup")
    print("=" * 60)
    print()

    install_packages()
    download_aact()
    ok = verify_files()

    print()
    if ok:
        print("✅  Setup complete!")
        print("    Next → python 01_clean.py")
    else:
        print("⚠️   Some files are missing.")
        print("    Please download manually from:")
        print("    https://aact.ctti-clinicaltrials.org/downloads")
