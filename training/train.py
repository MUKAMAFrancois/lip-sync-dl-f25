# training/train.py
import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import time

# --- SETUP PATHS ---
# We need access to the Wav2Lip model definitions
# Assuming Wav2Lip is cloned in the root or parallel
WAV2LIP_PATH = Path("../Wav2Lip") 
if not WAV2LIP_PATH.exists():
    WAV2LIP_PATH = Path("Wav2Lip")

sys.path.append(str(WAV2LIP_PATH))

try:
    from models import Wav2Lip, Wav2Lip_Disc_Qual
    import audio
except ImportError:
    print("❌ Critical Error: Could not import 'models' from Wav2Lip.")
    print("   Ensure you have cloned the repo: git clone https://github.com/Rudrabha/Wav2Lip.git")
    sys.exit(1)

# Import our local configs
from hparams import hparams
from data_loader import GermanDataset # We will create this next

# --- CONFIGURATION ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_DIR = Path("checkpoints")
LOG_DIR = Path("logs")
DATA_ROOT = Path("../data/german/preprocessed")
FILELIST_ROOT = Path("../data/german/filelists")

# Pre-trained GAN path (The "Foundation Model")
PRETRAINED_MODEL = "checkpoints/wav2lip_gan.pth"

def load_checkpoint(path, model, optimizer, reset_optimizer=False):
    print(f"🔄 Loading checkpoint: {path}")
    checkpoint = torch.load(path, map_location=DEVICE)
    
    # Fix for DataParallel keys if training happened on multi-GPU
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    
    model.load_state_dict(new_s)
    
    if not reset_optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    
    return model, optimizer, checkpoint.get("global_step", 0)

def save_checkpoint(model, optimizer, step, filename):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = CHECKPOINT_DIR / filename
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": step
    }, path)
    print(f"💾 Saved checkpoint: {path}")

def train():
    # 1. Setup Models
    print("🚀 Initializing Models...")
    model = Wav2Lip().to(DEVICE)
    disc = Wav2Lip_Disc_Qual().to(DEVICE) # The GAN Discriminator

    # 2. Setup Optimizers
    # Using lower learning rate for fine-tuning as per literature
    optimizer = optim.Adam(model.parameters(), lr=hparams['initial_learning_rate'] * 0.5)
    disc_optimizer = optim.Adam(disc.parameters(), lr=hparams['initial_learning_rate'] * 0.5)

    # 3. Load Pre-trained Weights (Fine-Tuning Mode)
    # We RESET the optimizer because we are starting a new task (German fine-tuning)
    if os.path.exists(PRETRAINED_MODEL):
        model, _, _ = load_checkpoint(PRETRAINED_MODEL, model, optimizer, reset_optimizer=True)
        # We generally don't load discriminator weights for new fine-tuning, 
        # allowing it to learn the specific "German" texture from scratch quickly.
    else:
        print("⚠️ Warning: No pre-trained model found! Training from scratch (Not Recommended).")

    # 4. Setup Data
    print("📂 Loading German Dataset...")
    train_ds = GermanDataset(DATA_ROOT, FILELIST_ROOT / "train.txt", hparams)
    test_ds = GermanDataset(DATA_ROOT, FILELIST_ROOT / "val.txt", hparams)
    
    train_loader = DataLoader(train_ds, batch_size=hparams['batch_size'], shuffle=True, num_workers=4)
    
    # 5. Loss Functions
    # L1 Loss for Pixel Accuracy
    recon_loss = nn.L1Loss()
    # Binary Cross Entropy for GAN (Real vs Fake)
    disc_loss = nn.BCELoss()

    # 6. Training Loop
    global_step = 0
    print(f"🎬 Starting Training on {DEVICE}...")
    
    model.train()
    disc.train()

    while global_step < hparams['nepochs']:
        for i, (input_frames, mel, gt_frames) in enumerate(tqdm(train_loader)):
            
            # Move data to GPU
            input_frames = input_frames.to(DEVICE) # [B, 6, 96, 96] (Masked + Reference)
            mel = mel.to(DEVICE)                   # [B, 1, 80, 16] (Audio)
            gt_frames = gt_frames.to(DEVICE)       # [B, 3, 96, 96] (Ground Truth)

            # --- A. Generator (Wav2Lip) Step ---
            optimizer.zero_grad()
            
            # Forward Pass
            generated_frames = model(input_frames, mel) # [B, 3, 96, 96]

            # 1. Perceptual Loss (Pixel Perfectness)
            l1 = recon_loss(generated_frames, gt_frames)
            
            # 2. Sync Loss (Lip Movement Accuracy)
            # (Requires the Expert SyncNet - usually frozen. For simplicity in this script, 
            # we focus on Visual Quality Fine-tuning. Full implementation requires loading SyncNet too.)
            
            # 3. GAN Loss (Realism)
            # The generator tries to fool the discriminator
            pred_fake = disc(generated_frames)
            gen_gan_loss = disc_loss(pred_fake, torch.ones_like(pred_fake))

            # Total Generator Loss
            # Weighted mix: Reconstruction (0.9) + GAN (0.07) + Sync (0.03)
            # We emphasize L1 for stability during fine-tuning
            total_loss = (hparams['syncnet_wt'] * 0.0) + \
                         ((1 - hparams['syncnet_wt'] - hparams['disc_wt']) * l1) + \
                         (hparams['disc_wt'] * gen_gan_loss)
            
            total_loss.backward()
            optimizer.step()

            # --- B. Discriminator Step ---
            disc_optimizer.zero_grad()

            pred_real = disc(gt_frames)
            pred_fake = disc(generated_frames.detach())

            loss_real = disc_loss(pred_real, torch.ones_like(pred_real))
            loss_fake = disc_loss(pred_fake, torch.zeros_like(pred_fake))
            
            d_loss = (loss_real + loss_fake) / 2
            d_loss.backward()
            disc_optimizer.step()

            global_step += 1

            # --- C. Logging & Saving ---
            if global_step % hparams['checkpoint_interval'] == 0:
                save_checkpoint(model, optimizer, global_step, f"german_finetune_step_{global_step}.pth")
                
                print(f"Step {global_step}: L1={l1.item():.4f}, GAN={gen_gan_loss.item():.4f}, Disc={d_loss.item():.4f}")

if __name__ == "__main__":
    train()