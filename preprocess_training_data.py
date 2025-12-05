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

SPLITS = {
    "train": "train",
    "valid": "val",
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
# Keep face detector on GPU, but process batches carefully
fa = FaceAlignment(LandmarksType._2D, flip_input=False, device=device)

def process_video(video_path, output_subdir):
    video_name = video_path.stem
    save_dir = output_subdir / video_name
    
    if save_dir.exists():
        # Check if it actually has data, otherwise overwrite
        if len(list(save_dir.glob("*.jpg"))) > 10:
            return [video_name]
        else:
            shutil.rmtree(save_dir)

    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Extract Audio
    audio_path = save_dir / "audio.wav"
    # -y overwrite, -vn no video, -ac 1 mono, 16k sample rate
    cmd = f'ffmpeg -y -v error -i "{video_path}" -ac 1 -vn -acodec pcm_s16le -ar 16000 "{audio_path}"'
    os.system(cmd)
    
    # 2. Open Video Stream
    cap = cv2.VideoCapture(str(video_path))
    
    # Memory Safe Batching
    batch_size = 16  # Process 16 frames at a time (Low RAM usage)
    valid_frames_count = 0
    global_frame_idx = 0 # Keeps track of frame number across batches
    
    while True:
        batch = []
        # Read a chunk of frames
        for _ in range(batch_size):
            ret, frame = cap.read()
            if not ret: break
            batch.append(frame)
            
        if not batch: break # End of video
        
        # 3. Detect & Crop (On this small batch only)
        try:
            # S3FD usually expects a numpy array of frames
            # batch_np shape: [Batch_Size, H, W, 3]
            preds = fa.get_detections_for_batch(np.array(batch))
        except Exception as e:
            # If detection fails on batch, skip it (rare)
            global_frame_idx += len(batch)
            continue
            
        for j, rect in enumerate(preds):
            if rect is None: 
                continue
            
            # DeepMind-Style Crop
            y1, y2, x1, x2 = rect
            cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
            h = y2 - y1
            size = int(h * 1.3)
            
            y_start = max(0, int(cy - size//2))
            y_end = min(batch[j].shape[0], int(cy + size//2))
            x_start = max(0, int(cx - size//2))
            x_end = min(batch[j].shape[1], int(cx + size//2))
            
            crop = batch[j][y_start:y_end, x_start:x_end]
            
            try:
                crop = cv2.resize(crop, (96, 96))
                
                # Use global index so file names are continuous: 00000, 00001, ...
                current_idx = global_frame_idx + j
                cv2.imwrite(str(save_dir / f"{current_idx:05d}.jpg"), crop)
                valid_frames_count += 1
            except: 
                pass
        
        # Increment counter by how many frames we read
        global_frame_idx += len(batch)
        
        # Python's Garbage Collector will now free 'batch' memory
        
    cap.release()
            
    return [video_name] if valid_frames_count > 10 else []

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(FILELISTS_DIR, exist_ok=True)
    
    for folder_name, split_name in SPLITS.items():
        input_split_dir = DATA_ROOT / folder_name
        if not input_split_dir.exists():
            print(f"⚠️ Warning: {folder_name} folder not found at {input_split_dir}")
            continue
            
        print(f"\n🎬 Processing split: {folder_name} -> {split_name}")
        videos = list(input_split_dir.glob("*.mp4"))
        
        manifest_lines = []
        
        for vid in tqdm(videos):
            results = process_video(vid, OUTPUT_ROOT)
            manifest_lines.extend(results)
            
        # Write Manifest
        with open(FILELISTS_DIR / f"{split_name}.txt", "w") as f:
            for line in manifest_lines:
                f.write(line + "\n")
                
    print("\n✅ Data Prep Complete!")

if __name__ == "__main__":
    main()