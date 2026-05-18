"""Download GUI-Net-1M and supplementary datasets."""
from __future__ import annotations
import subprocess
from pathlib import Path

RAW_DIR = Path("data/raw")


def download_all() -> None:
    download_gui_net_1m()


def download_gui_net_1m() -> None:
    out = RAW_DIR / "gui_net_1m"
    if out.exists():
        print(f"  GUI-Net-1M already at {out} — skipping")
        return
    out.mkdir(parents=True, exist_ok=True)
    print("  Downloading GUI-Net-1M from HuggingFace...")
    subprocess.run([
        "huggingface-cli", "download",
        "Bofeee5675/GUI-Net-1M",
        "--repo-type", "dataset",
        "--local-dir", str(out),
    ], check=True)
    print(f"  GUI-Net-1M saved to {out}")
