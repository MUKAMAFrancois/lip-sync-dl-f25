# preprocess_training_data.py
import sys
import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import torch
import shutil
import zipfile
import argparse

def check_cuda(force=False):
    """Verify CUDA availability"""
    print("\n" + "="*60)
    print("🔍 CUDA Environment Check")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"✅ CUDA Available: {torch.cuda.is_available()}")
        print(f"   GPU Device: {torch.cuda.get_device_name(0)}")
        return 'cuda'
    else:
        print("❌ CUDA NOT Available - Will use CPU.")
        if not force:
            print("   ⚠️ Warning: This will be very slow.")
        return 'cpu'

def setup_paths(kaggle_mode=False):
    if kaggle_mode:
        BASE = Path("/kaggle/working/lip-sync-dl-f25")
        INPUT_DATA = Path("/kaggle/input/muavic-german-sample/mtedx/video/de")
    else:
        BASE = Path(".")
        INPUT_DATA = Path("data/german")
    
    OUTPUT_ROOT = BASE / "data/german/preprocessed"
    FILELISTS_DIR = BASE / "data/german/filelists"
    
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(FILELISTS_DIR, exist_ok=True)
    
    return INPUT_DATA, OUTPUT_ROOT, FILELISTS_DIR, BASE

def setup_wav2lip(kaggle_mode=False):
    if kaggle_mode:
        WAV2LIP_PATH = Path("/kaggle/working/Wav2Lip")
    else:
        WAV2LIP_PATH = Path("Wav2Lip") 
        if not WAV2LIP_PATH.exists():
            WAV2LIP_PATH = Path("../Wav2Lip")
    
    if not WAV2LIP_PATH.exists():
        print("❌ Error: Wav2Lip not found! Run setup_wav2lip.py first.")
        sys.exit(1)
    
    if str(WAV2LIP_PATH) not in sys.path:
        sys.path.insert(0, str(WAV2LIP_PATH))
    
    try:
        from face_detection import FaceAlignment, LandmarksType
        return FaceAlignment, LandmarksType
    except ImportError:
        print("❌ Error: Wav2Lip face_detection not found.")
        sys.exit(1)

def compute_affine_transform(landmarks, target_size=256):
    # Standard Face Template (256x256)
    # - "face template in which the coordinates of these landmarks are fixed"
    canonical_points = np.float32([
        [70, 90],   # Left Eye
        [186, 90],  # Right Eye
        [128, 140]  # Nose Tip
    ]) * (target_size / 256.0)

    src_points = np.float32([
        np.mean(landmarks[36:42], axis=0), # Left Eye Mean
        np.mean(landmarks[42:48], axis=0), # Right Eye Mean
        landmarks[30]                      # Nose Tip
    ])

    # estimateAffinePartial2D handles scaling/rotation but no shear/skew
    M, _ = cv2.estimateAffinePartial2D(src_points, canonical_points)
    return M

def process_video(video_path, output_subdir, fa):
    video_name = video_path.stem
    save_dir = output_subdir / video_name
    
    if save_dir.exists():
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Extract Audio
    audio_path = save_dir / "audio.wav"
    cmd = f'ffmpeg -y -v error -i "{video_path}" -ac 1 -vn -acodec pcm_s16le -ar 16000 "{audio_path}"'
    os.system(cmd)
    
    cap = cv2.VideoCapture(str(video_path))
    batch_size = 4 # Conservative batch size
    valid_frames_count = 0
    global_frame_idx = 0 
    
    while True:
        batch = []
        for _ in range(batch_size):
            ret, frame = cap.read()
            if not ret: break
            batch.append(frame)
            
        if not batch: break 
        
        try:
            preds = fa.get_detections_for_batch(np.array(batch))
        except Exception:
            global_frame_idx += len(batch)
            continue
            
        for j, landmarks in enumerate(preds):
            if landmarks is None: continue
            
            try:
                # [cite: 134] Compute Transform
                M = compute_affine_transform(landmarks, target_size=256)
                if M is None: continue

                # [cite: 141] Warp Frame
                warped_face = cv2.warpAffine(
                    batch[j], M, (256, 256), flags=cv2.INTER_CUBIC
                )
                
                # Save Frame
                current_idx = global_frame_idx + j
                cv2.imwrite(str(save_dir / f"{current_idx:05d}.jpg"), warped_face)
                
                # --- CRITICAL FIX: Save Transformation Matrix ---
                # This is required for inference "Un-warping" 
                np.save(str(save_dir / f"{current_idx:05d}.npy"), M)
                
                valid_frames_count += 1
            except Exception: 
                pass
        
        global_frame_idx += len(batch)
        del batch
        
    cap.release()
    return [video_name] if valid_frames_count > 10 else []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kaggle', action='store_true')
    parser.add_argument('--no-zip', action='store_true')
    parser.add_argument('--force-cpu', action='store_true')
    args = parser.parse_args()
    
    device = check_cuda(args.force_cpu) if not args.force_cpu else 'cpu'
    
    # 1. Setup Paths
    INPUT_DATA, OUTPUT_ROOT, FILELISTS_DIR, BASE = setup_paths(args.kaggle)
    print(f"📂 Looking for data in: {INPUT_DATA.resolve()}")
    
    # 2. Setup Face Detector
    FaceAlignment, LandmarksType = setup_wav2lip(args.kaggle)
    print(f"🔥 Initializing Face Detector on {device.upper()}...")
    fa = FaceAlignment(LandmarksType._2D, flip_input=False, device=device)
    
    # 3. Process Splits (Added 'test' back)
    SPLITS = {
        "train": "train", 
        "val": "val", 
        "test": "test"
    }
    
    found_any = False
    for folder_name, split_name in SPLITS.items():
        input_split_dir = INPUT_DATA / folder_name
        
        # Explicit check with print
        if not input_split_dir.exists():
            print(f"⚠️ Skipping '{folder_name}': Directory not found at {input_split_dir}")
            continue
            
        found_any = True
        print(f"\n🎬 Processing: {folder_name} -> {split_name}")
        videos = list(input_split_dir.glob("*.mp4"))
        print(f"   Found {len(videos)} videos.")
        
        manifest_lines = []
        for vid in tqdm(videos):
            results = process_video(vid, OUTPUT_ROOT, fa)
            manifest_lines.extend(results)
            
        # Save filelist
        list_path = FILELISTS_DIR / f"{split_name}.txt"
        with open(list_path, "w") as f:
            for line in manifest_lines:
                f.write(line + "\n")
        print(f"   Saved list to {list_path}")
    
    if not found_any:
        print("\n❌ CRITICAL: No data folders found! Check your paths.")
        print(f"   Expected structure: {INPUT_DATA}/train/...")

    if not args.no_zip and found_any:
        print("📦 Zipping...")
        shutil.make_archive("german_preprocessed", 'zip', OUTPUT_ROOT)

if __name__ == "__main__":
    main()