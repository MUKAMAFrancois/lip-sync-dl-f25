import os
import subprocess
from pathlib import Path
import sys

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
            print("❌ Error cloning Wav2Lip.")
            return

    # 3. Download Pre-trained GAN Weights (The Foundation)
    # We use the GAN model because it produces sharper images than the standard model.
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    weights_path = CHECKPOINTS_DIR / "wav2lip_gan.pth"
    
    if not weights_path.exists():
        print(f"⬇️ Downloading GAN Weights to {weights_path}...")
        try:
            # Using curl for cross-platform compatibility (or wget via subprocess)
            url = "https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth"
            subprocess.run(["wget", "-q", "-O", str(weights_path), url], check=True)
            print("✅ Weights downloaded.")
        except Exception as e:
            print(f"❌ Error downloading weights: {e}")
            print("   Please download 'wav2lip_gan.pth' manually and place it in Wav2Lip/checkpoints/")
    else:
        print("✅ Weights already present.")

    # 4. Patching (Optional but Recommended)
    # If running on newer PyTorch, we might need to patch audio.py. 
    # For now, we assume requirements.txt handles versioning, but we can verify file existence.
    if (WAV2LIP_DIR / "audio.py").exists():
        print("✅ Wav2Lip structure verified.")

if __name__ == "__main__":
    setup()