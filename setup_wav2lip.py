import os
import subprocess
from pathlib import Path
import sys
import urllib.request  # <--- Standard Python library for downloading

def reporthook(blocknum, blocksize, totalsize):
    """
    Callback for urllib to show download progress.
    """
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(totalsize)), readsofar, totalsize)
        sys.stderr.write(s)
        if readsofar >= totalsize: # near the end
            sys.stderr.write("\n")
    else: # total size is unknown
        sys.stderr.write("read %d\n" % (readsofar,))

def setup():
    # 1. Define Root Path (Current Directory)
    ROOT_DIR = Path.cwd()
    WAV2LIP_DIR = ROOT_DIR / "Wav2Lip"
    CHECKPOINTS_DIR = WAV2LIP_DIR / "checkpoints"

    # 2. Clone Wav2Lip if missing
    if not WAV2LIP_DIR.exists():
        print(f"⬇️ Cloning Wav2Lip into {WAV2LIP_DIR}...")
        try:
            subprocess.run(["git", "clone", "https://github.com/Rudrabha/Wav2Lip.git"], check=True)
        except subprocess.CalledProcessError:
            print("❌ Error cloning Wav2Lip. Check git installation.")
            return

    # 3. Download Pre-trained GAN Weights (The Foundation)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    weights_path = CHECKPOINTS_DIR / "wav2lip_gan.pth"
    
    if not weights_path.exists():
        print(f"⬇️ Downloading GAN Weights to {weights_path}...")
        url = "https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth"
        
        try:
            # Cross-platform download using Python
            urllib.request.urlretrieve(url, str(weights_path), reporthook)
            print("✅ Weights downloaded successfully.")
        except Exception as e:
            print(f"\n❌ Error downloading weights: {e}")
            print("👉 Manual Fix: Download this link and put it in Wav2Lip/checkpoints/")
            print(f"   {url}")
    else:
        print("✅ Weights already present.")

    # 4. Verify Structure
    if (WAV2LIP_DIR / "audio.py").exists():
        print("✅ Wav2Lip structure verified.")

if __name__ == "__main__":
    setup()