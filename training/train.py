import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# --- ROBUST IMPORT ---
# Try to find Wav2Lip in cwd or parent
possible_paths = [Path("Wav2Lip"), Path("../Wav2Lip")]
for p in possible_paths:
    if p.exists():
        sys.path.append(str(p.resolve()))
        break

try:
    from models import Wav2Lip, Wav2Lip_Disc_Qual, SyncNet_color
except ImportError:
    print("❌ Critical: Could not import Wav2Lip models.")
    sys.exit(1)

# Import local modules (assumes running from project root)
sys.path.append(str(Path.cwd()))
try:
    from training.hparams import hparams
    from training.data_loader import GermanDataset
except ImportError:
    # Fallback if running directly inside training folder
    sys.path.append("..")
    from training.hparams import hparams
    from training.data_loader import GermanDataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_sync_loss(syncnet, mel, g):
    g = g[:, :, g.size(2)//2:, :] # Crop lower half
    g = torch.nn.functional.interpolate(g, size=(96, 96), mode='bilinear')
    a, v = syncnet(mel, g)
    return nn.CosineEmbeddingLoss()(a, v, torch.ones(a.size(0)).to(DEVICE))

def train():
    print(f"🚀 Training on {DEVICE}")
    
    # Paths (Hardcoded relative to project root for simplicity)
    DATA_ROOT = Path("data/german/preprocessed")
    FILELIST_ROOT = Path("data/german/filelists")
    CHECKPOINTS_DIR = Path("checkpoints")
    
    # 1. Models
    model = Wav2Lip().to(DEVICE)
    disc = Wav2Lip_Disc_Qual().to(DEVICE)
    syncnet = SyncNet_color().to(DEVICE)
    
    # Load Experts
    if not (CHECKPOINTS_DIR / "lipsync_expert.pth").exists():
        print("❌ Expert model missing. Run setup_wav2lip.py")
        sys.exit(1)

    sync_ckpt = torch.load("checkpoints/lipsync_expert.pth", map_location=DEVICE)
    syncnet.load_state_dict({k.replace('module.',''):v for k,v in sync_ckpt['state_dict'].items()})
    for p in syncnet.parameters(): p.requires_grad = False
    
    gan_ckpt_path = "checkpoints/wav2lip_gan.pth"
    if os.path.exists(gan_ckpt_path):
        print("🔄 Warm-starting from Pretrained GAN")
        gan_ckpt = torch.load(gan_ckpt_path, map_location=DEVICE)
        model.load_state_dict({k.replace('module.',''):v for k,v in gan_ckpt['state_dict'].items()}, strict=False)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    disc_optimizer = optim.Adam(disc.parameters(), lr=1e-4)
    scaler = GradScaler()

    # 2. Data
    if not (FILELIST_ROOT / "train.txt").exists():
        print("❌ Train list not found. Run preprocessing first.")
        return

    train_ds = GermanDataset(DATA_ROOT, FILELIST_ROOT / "train.txt", hparams)
    train_loader = DataLoader(train_ds, batch_size=hparams['batch_size'], shuffle=True, num_workers=0)

    # 3. Loop
    global_step = 0
    recon_loss = nn.L1Loss()
    disc_loss = nn.BCELoss()

    model.train()
    print("🎬 Starting Loop...")
    
    while global_step < 10000: # Short test run limit
        for input_frames, mel, gt_frames in tqdm(train_loader):
            input_frames, mel, gt_frames = input_frames.to(DEVICE), mel.to(DEVICE), gt_frames.to(DEVICE)

            # Generator
            with autocast():
                gen_frames = model(input_frames, mel)
                l1 = recon_loss(gen_frames, gt_frames)
                sync = get_sync_loss(syncnet, mel, gen_frames)
                fake_pred = disc(gen_frames)
                gan = disc_loss(fake_pred, torch.ones_like(fake_pred))
                loss = 0.03*sync + 0.9*l1 + 0.07*gan
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Discriminator
            with autocast():
                real_pred = disc(gt_frames)
                fake_pred = disc(gen_frames.detach())
                d_loss = (disc_loss(real_pred, torch.ones_like(real_pred)) + 
                          disc_loss(fake_pred, torch.zeros_like(fake_pred))) / 2
            
            scaler.scale(d_loss).backward()
            scaler.step(disc_optimizer)
            scaler.update()
            disc_optimizer.zero_grad()

            global_step += 1
            if global_step % 50 == 0:
                print(f"Step {global_step} | L1: {l1.item():.4f}")

if __name__ == "__main__":
    train()