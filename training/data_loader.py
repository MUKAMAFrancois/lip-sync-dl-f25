# training/data_loader.py
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import random
from pathlib import Path

# We need the audio processor from Wav2Lip
# (Ensure Wav2Lip is in your python path or sys.path)
try:
    import audio 
except ImportError:
    # Fallback if running from a different directory structure
    import sys
    sys.path.append("Wav2Lip")
    import audio

class GermanDataset(Dataset):
    def __init__(self, data_root, filelist_path, hparams):
        self.data_root = Path(data_root)
        self.hparams = hparams
        self.img_size = hparams['img_size'] # Dynamic Resolution (256)
        
        # Load Video List
        with open(filelist_path, 'r') as f:
            self.video_names = [x.strip() for x in f.readlines()]
            
        self.all_data = []
        # Pre-scan videos to index valid frames
        print(f"Indexing dataset (Resolution: {self.img_size}x{self.img_size})...")
        for vid in self.video_names:
            vid_path = self.data_root / vid
            frames = sorted(list(vid_path.glob("*.jpg")))
            
            # We need at least 5 frames window to train (Wav2Lip context)
            if len(frames) > 5:
                # Store (VideoName, FrameIndex)
                # We index the MIDDLE frame of a 5-frame window
                for i in range(2, len(frames) - 2):
                    self.all_data.append((vid, i))
                    
        print(f"Dataset Loaded: {len(self.all_data)} samples.")

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        vid_name, frame_idx = self.all_data[idx]
        vid_path = self.data_root / vid_name
        
        # 1. Load Window of 5 Frames
        window_frames = []
        window_indices = range(frame_idx - 2, frame_idx + 3) # [-2, -1, 0, +1, +2]
        
        for i in window_indices:
            img_path = vid_path / f"{i:05d}.jpg"
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            # CRITICAL FIX: Use hparams resolution (256), don't hardcode 96!
            img = cv2.resize(img, (self.img_size, self.img_size))
            window_frames.append(img)

        # Safety check if frames are missing
        if len(window_frames) < 5:
            # Recursively try another random sample if this one is broken
            return self.__getitem__(random.randint(0, len(self.all_data) - 1))

        # 2. Masking Strategy (Paper Section 3.1.2) [cite: 164]
        # We mask the lower half of the TARGET frame.
        target_img = window_frames[2] # Center frame
        masked_img = target_img.copy()
        
        # DeepMind Paper suggests a polygon mask, but mentions a rectangular mask 
        # is "sufficient" for the input[cite: 309]. 
        # We mask the bottom half to force the model to look at context + audio.
        masked_img[self.img_size // 2:] = 0 
        
        # 3. Reference Frame Selection (Paper Section 3.1.4) [cite: 171]
        # The paper uses K-Means to pick "diverse" frames. 
        # For this baseline, we stick to Random Sampling but ensure it's from the same video.
        all_frames = list(vid_path.glob("*.jpg"))
        rand_idx = random.choice(range(len(all_frames)))
        ref_img_path = vid_path / f"{rand_idx:05d}.jpg"
        ref_img = cv2.imread(str(ref_img_path))
        if ref_img is None: ref_img = target_img
        ref_img = cv2.resize(ref_img, (self.img_size, self.img_size))

        # 4. Audio Processing (Paper Section 3.1.3) [cite: 254]
        # The paper uses a Mel-spectrogram with 80 filter banks.
        # We assume 'audio.wav' exists in the folder (created by preprocess script).
        audio_path = vid_path / "audio.wav"
        
        # Wav2Lip's audio.py handles the Mel-spectrogram generation
        # It needs the full audio and the specific time step
        try:
            # Calculate start time of this frame (assuming 25 FPS for simplicity, or 30)
            # Better: load the whole wav and slice.
            # Standard Wav2Lip logic:
            fps = 25 
            mel = audio.melspectrogram(audio.load_wav(audio_path, 16000))
            
            # Identify the Mel slice corresponding to the video frame
            mel_step_size = 16 
            mel_chunks = len(mel[0])
            vid_chunks = len(all_frames)
            
            # Map video frame index to audio Mel index
            mel_idx_multiplier = mel_chunks / vid_chunks
            i = int(frame_idx * mel_idx_multiplier)
            
            # Wav2Lip expects 16 mel steps (approx 0.2s of audio)
            if i < mel_step_size: 
                i = mel_step_size
            if i >= mel_chunks - mel_step_size:
                i = mel_chunks - mel_step_size - 1
                
            mel_slice = mel[:, i - mel_step_size : i] # [80, 16]
        except Exception as e:
            # print(f"Audio Error: {e}")
            # Fallback to silence if audio fails (prevents crash)
            mel_slice = torch.zeros(80, 16)

        # 5. Prepare Tensors
        # Concatenate: Masked (3ch) + Reference (3ch) = 6 Channels
        input_tensor = np.concatenate([masked_img, ref_img], axis=2) 
        
        # Normalize to [0, 1] then transpose to [C, H, W]
        input_tensor = input_tensor.transpose(2, 0, 1) / 255.0       
        gt_tensor = target_img.transpose(2, 0, 1) / 255.0  
        
        # Return: Input [6, H, W], Audio [1, 80, 16], GroundTruth [3, H, W]
        return (torch.FloatTensor(input_tensor), 
                torch.FloatTensor(mel_slice).unsqueeze(0), 
                torch.FloatTensor(gt_tensor))