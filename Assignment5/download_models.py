"""
Download ViT-S model files using huggingface_hub (no PyTorch needed on host).
Run this OUTSIDE Docker:  python3 download_models.py
"""
import os

try:
    from huggingface_hub import snapshot_download
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "huggingface_hub", "--break-system-packages", "--quiet"])
    from huggingface_hub import snapshot_download

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vit_small_local")
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"Downloading WinKawaks/vit-small-patch16-224 → {SAVE_DIR} ...")
path = snapshot_download(
    repo_id="WinKawaks/vit-small-patch16-224",
    local_dir=SAVE_DIR,
    local_dir_use_symlinks=False,
)
print(f"\nDone! Files saved to: {path}")
print("Files:", os.listdir(SAVE_DIR))
