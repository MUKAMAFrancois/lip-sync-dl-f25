import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import random
from pathlib import Path

class GermanDataset(Dataset):
    def __init__(self, data_root, filelist_path, hparams):
        self.data_root = Path(data_root)
        self.hparams = hparams
        
        # Load Video List
        with open(filelist_path, 'r') as f:
            self.video_names = [x.strip() for x in f.readlines()]
            
        self.all_data = []
        # Pre-scan videos to index valid frames (Crucial for speed)
        for vid in self.video_names:
            vid_path = self.data_root / vid
            frames = sorted(list(vid_path.glob("*.jpg")))
            
            # We need at least 5 frames window to train
            if len(frames) > 5:
                # Store (VideoName, FrameIndex)
                # Wav2Lip takes a window of 5 frames. We index the MIDDLE frame.
                for i in range(2, len(frames) - 2):
                    self.all_data.append((vid, i))

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        vid_name, frame_idx = self.all_data[idx]
        vid_path = self.data_root / vid_name
        
        # 1. Load Window of 5 Frames
        # Why 5? To capture temporal motion (jaw moving down/up)
        window_frames = []
        window_indices = range(frame_idx - 2, frame_idx + 3) # [-2, -1, 0, +1, +2]
        
        for i in window_indices:
            img_path = vid_path / f"{i:05d}.jpg"
            img = cv2.imread(str(img_path))
            if img is None:
                # Fallback: just use center frame (should rarely happen)
                img_path = vid_path / f"{frame_idx:05d}.jpg"
                img = cv2.imread(str(img_path))
            
            img = cv2.resize(img, (96, 96))
            window_frames.append(img)

        # 2. Masking (The "DeepMind" Strategy) [cite: 163]
        # We mask the lower half of the INPUT frames so the model *must* use audio to guess.
        # But we keep the GROUND TRUTH frames unmasked to calculate loss.
        
        # Concatenate 5 frames along channel dimension
        # Input: [Masked_Window (15 channels)] + [Reference (3 channels)]?
        # Standard Wav2Lip Input: [Masked_Target (3)] + [Reference (3)]
        # Actually, standard Wav2Lip uses:
        # Input: [Masked_Current_Frame, Reference_Frame] -> 6 Channels
        
        # Let's implement the standard Wav2Lip Input format for compatibility:
        # Target Frame
        target_img = window_frames[2] # Center frame
        
        # Mask lower half
        masked_img = target_img.copy()
        masked_img[96//2:] = 0 # Black out mouth area
        
        # Reference Frame (Random frame from same video)
        # This gives identity/texture info [cite: 170]
        rand_idx = random.choice(range(len(list(vid_path.glob("*.jpg")))))
        ref_img_path = vid_path / f"{rand_idx:05d}.jpg"
        ref_img = cv2.imread(str(ref_img_path))
        if ref_img is None: ref_img = target_img
        ref_img = cv2.resize(ref_img, (96, 96))

        # 3. Load Audio (Mel Spectrogram)
        # We need the audio slice corresponding exactly to the target frame
        # (This usually requires 'audio.py' from Wav2Lip to slice the Mel)
        # For simplicity in this loader, we return a dummy or implement basic Mel slicing:
        # NOTE: In production, you MUST use the Wav2Lip audio processor.
        # We will assume a placeholder tensor here to let the script run, 
        # but you should import 'audio' from Wav2Lip path.
        
        # Placeholder Mel (1 channel, 80 bands, 16 time steps)
        mel = torch.randn(1, 80, 16) 

        # 4. Prepare Tensors
        # Normalize to [0, 1] then [-1, 1] implies standard GAN normalization,
        # but Wav2Lip usually expects [0, 1].
        
        # Stack: Masked (3ch) + Reference (3ch) = 6 Channels
        input_tensor = np.concatenate([masked_img, ref_img], axis=2) # [96, 96, 6]
        input_tensor = input_tensor.transpose(2, 0, 1) / 255.0       # [6, 96, 96]
        
        gt_tensor = target_img.transpose(2, 0, 1) / 255.0            # [3, 96, 96]
        
        return torch.FloatTensor(input_tensor), mel, torch.FloatTensor(gt_tensor)