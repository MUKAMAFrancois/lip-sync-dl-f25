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
from torch.cuda.amp import autocast, GradScaler #

# --- SETUP PATHS ---
WAV2LIP_PATH = Path("../Wav2Lip") 
if not WAV2LIP_PATH.exists():
    WAV2LIP_PATH = Path("Wav2Lip")
sys.path.append(str(WAV2LIP_PATH))

try:
    from models import Wav2Lip, Wav2Lip_Disc_Qual, SyncNet_color
except ImportError:
    print("❌ Error: Wav2Lip models not found.")
    sys.exit(1)

from hparams import hparams
from data_loader import GermanDataset

# --- PERFORMANCE CONFIG ---
DEVICE = 'cuda'
ACCUMULATION_STEPS = 8 # Effective Batch = batch_size * 8 = 64 (Matches Paper)
USE_AMP = True         # Mixed Precision for 2x Speed

CHECKPOINT_DIR = Path("checkpoints")
DATA_ROOT = Path("data/german/preprocessed")
FILELIST_ROOT = Path("data/german/filelists")
SYNC_EXPERT_PATH = CHECKPOINT_DIR / "lipsync_expert.pth"
GAN_PRETRAINED_PATH = CHECKPOINT_DIR / "wav2lip_gan.pth"

def load_checkpoint(path, model, optimizer=None, reset_optimizer=False):
    print(f"🔄 Loading: {path}")
    checkpoint = torch.load(path, map_location=DEVICE)
    s = checkpoint["state_dict"]
    new_s = {k.replace('module.', ''): v for k, v in s.items()}
    model.load_state_dict(new_s, strict=False)
    if optimizer and not reset_optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer, checkpoint.get("global_step", 0)

def save_checkpoint(model, optimizer, step, filename):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": step
    }, CHECKPOINT_DIR / filename)

def get_sync_loss(syncnet, mel, g):
    # Crop lower half for SyncNet
    g = g[:, :, g.size(2)//2:, :] 
    # Resize to 96x96 (SyncNet native resolution)
    g = torch.nn.functional.interpolate(g, size=(96, 96), mode='bilinear', align_corners=False)
    a, v = syncnet(mel, g)
    loss = nn.CosineEmbeddingLoss()(a, v, torch.ones(a.size(0)).to(DEVICE))
    return loss

def train():
    print("🚀 Initializing High-Performance Training...")
    
    # 1. Models
    model = Wav2Lip().to(DEVICE)
    disc = Wav2Lip_Disc_Qual().to(DEVICE)
    
    # Load Frozen SyncNet
    if not SYNC_EXPERT_PATH.exists():
        print("❌ lipsync_expert.pth missing!")
        sys.exit(1)
    syncnet = SyncNet_color().to(DEVICE)
    load_checkpoint(SYNC_EXPERT_PATH, syncnet)
    for p in syncnet.parameters(): p.requires_grad = False
    syncnet.eval()

    # 2. Optimization
    # Paper uses specific LRs. We stick to hparams but scaled for fine-tuning.
    optimizer = optim.Adam(model.parameters(), lr=hparams['initial_learning_rate'] * 0.5)
    disc_optimizer = optim.Adam(disc.parameters(), lr=hparams['initial_learning_rate'] * 0.5)
    
    # AMP Scaler
    scaler = GradScaler(enabled=USE_AMP)
    disc_scaler = GradScaler(enabled=USE_AMP)

    # 3. Load Pretrained Weights
    if GAN_PRETRAINED_PATH.exists():
        load_checkpoint(GAN_PRETRAINED_PATH, model, optimizer, reset_optimizer=True)
    
    # 4. Data
    # For Speaker-Specific Speed run: Ensure train.txt only contains YOUR TARGET VIDEO
    train_ds = GermanDataset(DATA_ROOT, FILELIST_ROOT / "train.txt", hparams)
    train_loader = DataLoader(train_ds, batch_size=hparams['batch_size'], shuffle=True, num_workers=4)
    
    recon_loss = nn.L1Loss()
    disc_loss = nn.BCELoss()
    
    global_step = 0
    print(f"🎬 Starting Training (AMP={USE_AMP}, Accumulation={ACCUMULATION_STEPS})...")
    
    model.train()
    disc.train()

    optimizer.zero_grad()
    disc_optimizer.zero_grad()

    while global_step < hparams['nepochs']:
        for i, (input_frames, mel, gt_frames) in enumerate(tqdm(train_loader)):
            
            # Data to GPU
            input_frames = input_frames.to(DEVICE, non_blocking=True)
            mel = mel.to(DEVICE, non_blocking=True)
            gt_frames = gt_frames.to(DEVICE, non_blocking=True)

            # ==========================
            # 1. Train Generator
            # ==========================
            with autocast(enabled=USE_AMP):
                generated_frames = model(input_frames, mel)
                
                # Losses
                l1 = recon_loss(generated_frames, gt_frames)
                sync_loss = get_sync_loss(syncnet, mel, generated_frames)
                pred_fake = disc(generated_frames)
                gen_gan_loss = disc_loss(pred_fake, torch.ones_like(pred_fake))
                
                # Weighted Sum (Matches Paper Logic [cite: 358])
                # Note: We scale loss by Accumulation Steps
                total_loss = ((0.03 * sync_loss) + (0.9 * l1) + (0.07 * gen_gan_loss)) / ACCUMULATION_STEPS

            # Backward (Scaled)
            scaler.scale(total_loss).backward()

            # Step (only every N batches)
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # ==========================
            # 2. Train Discriminator
            # ==========================
            with autocast(enabled=USE_AMP):
                pred_real = disc(gt_frames)
                pred_fake = disc(generated_frames.detach()) # Detach to avoid backprop to G
                
                loss_real = disc_loss(pred_real, torch.ones_like(pred_real))
                loss_fake = disc_loss(pred_fake, torch.zeros_like(pred_fake))
                d_loss = ((loss_real + loss_fake) / 2) / ACCUMULATION_STEPS

            disc_scaler.scale(d_loss).backward()

            if (i + 1) % ACCUMULATION_STEPS == 0:
                disc_scaler.step(disc_optimizer)
                disc_scaler.update()
                disc_optimizer.zero_grad()
                global_step += 1 # Only count actual updates
                
                # Logging
                if global_step % 100 == 0: # Log frequent updates
                    print(f" Step {global_step}: L1={l1.item():.4f} Sync={sync_loss.item():.4f}")

            # Checkpointing
            if global_step % hparams['checkpoint_interval'] == 0 and global_step > 0:
                save_checkpoint(model, optimizer, global_step, f"german_fast_step_{global_step}.pth")
                
                # Early Exit Strategy: If L1 is very low, we can stop early
                if l1.item() < 0.035:
                    print("🎯 Target L1 reached. Stopping early.")
                    save_checkpoint(model, optimizer, global_step, "german_final.pth")
                    return

if __name__ == "__main__":
    train()