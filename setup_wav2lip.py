import os
import subprocess
from pathlib import Path
import sys
import urllib.request

def download_file(url, dest_path):
    if dest_path.exists():
        print(f" Found: {dest_path.name}")
        return
    print(f" Downloading {dest_path.name}...")
    try:
        urllib.request.urlretrieve(url, str(dest_path))
    except Exception as e:
        print(f"!!! Error: {e}")

def setup():
    PROJECT_ROOT = Path.cwd()
    WAV2LIP_DIR = PROJECT_ROOT / "Wav2Lip"
    CHECKPOINTS_DIR = Path("checkpoints")

    if not WAV2LIP_DIR.exists():
        print(f" Cloning Wav2Lip into {WAV2LIP_DIR}...")
        subprocess.run(["git", "clone", "https://github.com/Rudrabha/Wav2Lip.git"], check=True)
    
    (WAV2LIP_DIR / "__init__.py").touch()
    
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    models = {
        "wav2lip_gan.pth": "https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth",
        "lipsync_expert.pth": "https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/lipsync_expert.pth"
    }
    for name, url in models.items():
        download_file(url, CHECKPOINTS_DIR / name)

    print(" Setup Complete.")

if __name__ == "__main__":
    setup()