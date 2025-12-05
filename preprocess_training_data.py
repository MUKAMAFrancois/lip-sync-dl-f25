import sys
import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import torch
import shutil

# --- CONFIGURATION ---
DATA_ROOT = Path("data/german")
OUTPUT_ROOT = Path("data/german/preprocessed")
FILELISTS_DIR = Path("data/german/filelists")

# Map your folders to Wav2Lip splits
SPLITS = {
    "train": "train",
    "valid": "val",  # Wav2Lip standard name is 'val'
    "test": "test"
}

# --- WAV2LIP SETUP ---
WAV2LIP_PATH = Path("Wav2Lip") 
if not WAV2LIP_PATH.exists():
    WAV2LIP_PATH = Path("../Wav2Lip") 

sys.path.append(str(WAV2LIP_PATH))

try:
    from face_detection import FaceAlignment, LandmarksType
except ImportError:
    print("❌ Error: Wav2Lip face_detection not found. Clone Wav2Lip first.")
    sys.exit(1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
fa = FaceAlignment(LandmarksType._2D, flip_input=False, device=device)

def process_video(video_path, output_subdir):
    video_name = video_path.stem
    save_dir = output_subdir / video_name
    
    if save_dir.exists():
        return [video_name] # Skip if done

    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Extract Audio
    audio_path = save_dir / "audio.wav"
    cmd = f'ffmpeg -y -i "{video_path}" -ac 1 -vn -acodec pcm_s16le -ar 16000 "{audio_path}" -loglevel quiet'
    os.system(cmd)
    
    # 2. Read Frames
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    
    if not frames: return []

    # 3. Detect & Crop
    batch_size = 4
    valid_frames = 0
    
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
        try:
            preds = fa.get_detections_for_batch(np.array(batch))
        except: continue
            
        for j, rect in enumerate(preds):
            if rect is None: continue
            
            # DeepMind-Style Crop: Focus on lower face
            y1, y2, x1, x2 = rect
            cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
            h = y2 - y1
            size = int(h * 1.3) # 1.3x zoom captures chin to nose perfectly
            
            y_start = max(0, int(cy - size//2))
            y_end = min(batch[j].shape[0], int(cy + size//2))
            x_start = max(0, int(cx - size//2))
            x_end = min(batch[j].shape[1], int(cx + size//2))
            
            crop = batch[j][y_start:y_end, x_start:x_end]
            
            try:
                crop = cv2.resize(crop, (96, 96)) # Standard Wav2Lip Input
                cv2.imwrite(str(save_dir / f"{i+j:05d}.jpg"), crop)
                valid_frames += 1
            except: pass
            
    return [video_name] if valid_frames > 10 else []

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(FILELISTS_DIR, exist_ok=True)
    
    for folder_name, split_name in SPLITS.items():
        input_split_dir = DATA_ROOT / folder_name
        if not input_split_dir.exists():
            print(f"⚠️ Warning: {folder_name} folder not found.")
            continue
            
        print(f"\n🎬 Processing split: {folder_name} -> {split_name}")
        videos = list(input_split_dir.glob("*.mp4"))
        
        manifest_lines = []
        
        for vid in tqdm(videos):
            # We save everything into one big preprocessed folder, 
            # but the manifest tells train.py which is which.
            results = process_video(vid, OUTPUT_ROOT)
            manifest_lines.extend(results)
            
        # Write Manifest (train.txt / val.txt / test.txt)
        with open(FILELISTS_DIR / f"{split_name}.txt", "w") as f:
            for line in manifest_lines:
                f.write(line + "\n")
                
    print("\n✅ Data Prep Complete!")

if __name__ == "__main__":
    main()