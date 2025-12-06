# setup_wav2lip.py
import os
import subprocess
from pathlib import Path
import sys
import urllib.request

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

def download_file(url, dest_path):
    if dest_path.exists():
        print(f"✅ Found: {dest_path.name}")
        return

    print(f"⬇️ Downloading {dest_path.name}...")
    try:
        urllib.request.urlretrieve(url, str(dest_path), reporthook)
        print(f"✅ Download complete: {dest_path.name}")
    except Exception as e:
        print(f"\n❌ Error downloading {dest_path.name}: {e}")
        print(f"👉 Manual Link: {url}")

def setup():
    # 1. Define Paths
    # CRITICAL CHANGE: We go UP one level (.parent) to place Wav2Lip next to our repo,
    # not inside it.
    PROJECT_ROOT = Path.cwd() 
    WAV2LIP_DIR = PROJECT_ROOT.parent / "Wav2Lip"
    CHECKPOINTS_DIR = WAV2LIP_DIR / "checkpoints"

    print(f"📂 Setting up Wav2Lip at: {WAV2LIP_DIR}")

    # 2. Clone Wav2Lip if missing
    if not WAV2LIP_DIR.exists():
        print(f"⬇️ Cloning Wav2Lip...")
        try:
            # We explicitly run git clone in the parent directory
            subprocess.run(
                ["git", "clone", "https://github.com/Rudrabha/Wav2Lip.git"], 
                cwd=PROJECT_ROOT.parent, # <--- Execute command in the parent folder
                check=True
            )
        except subprocess.CalledProcessError:
            print("❌ Error cloning Wav2Lip. Check git installation.")
            return
    else:
        print("✅ Wav2Lip repo already exists.")

    # 3. Download Pre-trained Weights (GAN + Expert)
    # These are needed for fine-tuning.
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    
    models_to_download = {
        "wav2lip_gan.pth": "https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth",
        "lipsync_expert.pth": "https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/lipsync_expert.pth"
    }

    print("\n--- Checking Checkpoints ---")
    for filename, url in models_to_download.items():
        download_file(url, CHECKPOINTS_DIR / filename)

    # 4. Verify Structure
    if (WAV2LIP_DIR / "audio.py").exists():
        print("\n✅ Wav2Lip structure verified.")
        print(f"   path: {WAV2LIP_DIR}")

if __name__ == "__main__":
    setup()