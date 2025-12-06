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

def check_cuda():
    """Verify CUDA availability and print GPU info"""
    print("\n" + "="*60)
    print("🔍 CUDA Environment Check")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"✅ CUDA Available: {torch.cuda.is_available()}")
        print(f"   GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"   Current Device: cuda:{torch.cuda.current_device()}")
        
        # Force CUDA operations to fail fast if there's an issue
        torch.cuda.init()
        return 'cuda'
    else:
        print("❌ CUDA NOT Available - Will use CPU (VERY SLOW)")
        print("   ⚠️ Warning: Processing will take 10-50x longer on CPU")
        response = input("   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            sys.exit(0)
        return 'cpu'

def setup_paths(kaggle_mode=False):
    """Configure paths based on environment"""
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
    """Setup Wav2Lip dependencies"""
    if kaggle_mode:
        # Wav2Lip is downloaded via setup_wav2lip.py to /kaggle/working/Wav2Lip
        WAV2LIP_PATH = Path("/kaggle/working/Wav2Lip")
    else:
        WAV2LIP_PATH = Path("Wav2Lip") 
        if not WAV2LIP_PATH.exists():
            WAV2LIP_PATH = Path("../Wav2Lip")
    
    if not WAV2LIP_PATH.exists():
        print("❌ Error: Wav2Lip not found!")
        if kaggle_mode:
            print("   Please run: !python setup_wav2lip.py first")
        else:
            print("   Please run: python setup_wav2lip.py first")
        sys.exit(1)
    
    if str(WAV2LIP_PATH) not in sys.path:
        sys.path.insert(0, str(WAV2LIP_PATH))
    
    # Clean up any conflicting face_detection packages
    print("🧹 Removing conflicting packages...")
    os.system("pip uninstall -y face-detection 2>/dev/null")
    
    try:
        from face_detection import FaceAlignment, LandmarksType
        print(f"✅ Wav2Lip loaded from: {WAV2LIP_PATH}")
        return FaceAlignment, LandmarksType
    except ImportError:
        print("❌ Error: Wav2Lip face_detection not found.")
        print("   Ensure face-alignment is installed:")
        print("   pip install face-alignment")
        sys.exit(1)

# --- PAPER IMPLEMENTATION: View Canonicalization ---
# Reference: Section 3.1.1 "View Canonicalization" 
def compute_affine_transform(landmarks, target_size=256):
    """
    Calculates the Affine Matrix to align eyes and nose to a standard template.
    This stabilizes the face and removes head rotation.
    """
    # 1. Define Canonical Points (Standard Face Template) for 256x256
    # These coordinates "lock" the face in the center of the image.
    canonical_points = np.float32([
        [70, 90],   # Left Eye Center
        [186, 90],  # Right Eye Center
        [128, 140]  # Nose Tip
    ]) * (target_size / 256.0) # Scale invariance

    # 2. Extract specific landmarks from the 68-point set (provided by FaceAlignment)
    # Indices: 36-41 (L Eye), 42-47 (R Eye), 30 (Nose Tip)
    src_points = np.float32([
        np.mean(landmarks[36:42], axis=0), # Left Eye Mean
        np.mean(landmarks[42:48], axis=0), # Right Eye Mean
        landmarks[30]                      # Nose Tip
    ])

    # 3. Calculate Affine Matrix (Scale + Rotation + Translation)
    # estimateAffinePartial2D handles scaling and rotation but avoids skew/shear
    M, _ = cv2.estimateAffinePartial2D(src_points, canonical_points)
    
    return M

def process_video(video_path, output_subdir, fa):
    """Process a single video: extract audio and face crops"""
    video_name = video_path.stem
    save_dir = output_subdir / video_name
    
    if save_dir.exists():
        if len(list(save_dir.glob("*.jpg"))) > 10:
            return [video_name]
        shutil.rmtree(save_dir)

    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Extract Audio
    audio_path = save_dir / "audio.wav"
    cmd = f'ffmpeg -y -v error -i "{video_path}" -ac 1 -vn -acodec pcm_s16le -ar 16000 "{audio_path}"'
    os.system(cmd)
    
    # 2. Open Video Stream
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Optimize batch size based on GPU memory
    batch_size = 8  #
    valid_frames_count = 0
    global_frame_idx = 0 
    
    pbar = tqdm(total=total_frames, desc=f"   {video_name}", leave=False, unit="fr")
    
    while True:
        batch = []
        for _ in range(batch_size):
            ret, frame = cap.read()
            if not ret: break
            batch.append(frame)
            
        if not batch: break 
        
        try:
            # This runs on GPU if fa.device == 'cuda'
            preds = fa.get_detections_for_batch(np.array(batch))
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n⚠️ GPU OOM! Reduce batch_size from {batch_size}")
                torch.cuda.empty_cache()
            pbar.update(len(batch))
            global_frame_idx += len(batch)
            continue
        except Exception:
            pbar.update(len(batch))
            global_frame_idx += len(batch)
            continue
            
        for j, landmarks in enumerate(preds):
            if landmarks is None: 
                continue
            
            # --- PAPER IMPLEMENTATION START ---
            try:
                # 1. Compute Affine Matrix (Stabilization) [cite: 134]
                M = compute_affine_transform(landmarks, target_size=256)
                
                if M is None: continue

                # 2. Warp the frame (Bicubic Resampling) [cite: 141]
                # This aligns the face to 256x256
                warped_face = cv2.warpAffine(
                    batch[j], M, (256, 256), flags=cv2.INTER_CUBIC
                )
                
                # 3. Save Frame
                current_idx = global_frame_idx + j
                cv2.imwrite(str(save_dir / f"{current_idx:05d}.jpg"), warped_face)
                valid_frames_count += 1
            except Exception as e: 
                # print(f"Skipping frame: {e}")
                pass
            # --- PAPER IMPLEMENTATION END ---
        
        pbar.update(len(batch))
        global_frame_idx += len(batch)
        del batch
        
        # Clear GPU cache periodically
        if global_frame_idx % 256 == 0:
            torch.cuda.empty_cache()
        
    pbar.close()
    cap.release()
            
    return [video_name] if valid_frames_count > 10 else []

def create_archive(base_path, output_name="german_preprocessed_data.zip"):
    """Create downloadable ZIP archive"""
    print(f"\n📦 Creating archive: {output_name}")
    archive_path = base_path / output_name
    
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        data_dir = base_path / "data/german"
        
        for root, dirs, files in os.walk(data_dir / "preprocessed"):
            for file in tqdm(files, desc="Zipping", leave=False):
                file_path = Path(root) / file
                arcname = str(file_path.relative_to(base_path))
                zipf.write(file_path, arcname)
        
        # Include filelists
        for file in (data_dir / "filelists").glob("*.txt"):
            arcname = str(file.relative_to(base_path))
            zipf.write(file, arcname)
    
    size_mb = archive_path.stat().st_size / (1024**2)
    print(f"✅ Archive created: {archive_path.name} ({size_mb:.1f} MB)")
    return archive_path

def main():
    parser = argparse.ArgumentParser(description='Preprocess German video data for Wav2Lip training')
    parser.add_argument('--kaggle', action='store_true', help='Run in Kaggle mode')
    parser.add_argument('--no-zip', action='store_true', help='Skip creating ZIP archive')
    parser.add_argument('--force-cpu', action='store_true', help='Force CPU processing (not recommended)')
    args = parser.parse_args()
    
    # 0. Check CUDA first
    device = check_cuda() if not args.force_cpu else 'cpu'
    
    # 1. Setup Environment
    print(f"\n🚀 Running in {'KAGGLE' if args.kaggle else 'LOCAL'} mode")
    INPUT_DATA, OUTPUT_ROOT, FILELISTS_DIR, BASE = setup_paths(args.kaggle)
    
    # 2. Setup Wav2Lip
    FaceAlignment, LandmarksType = setup_wav2lip(args.kaggle)
    
    print(f"\n🔥 Initializing Face Detector on {device.upper()}...")
    fa = FaceAlignment(LandmarksType._2D, flip_input=False, device=device)
    
    # Verify it's actually on GPU
    if device == 'cuda':
        print(f"✅ Face detector loaded on GPU")
        print(f"   Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # 3. Process Videos
    SPLITS = {"train": "train", "valid": "val", "test": "test"}
    
    for folder_name, split_name in SPLITS.items():
        input_split_dir = INPUT_DATA / folder_name
        if not input_split_dir.exists():
            print(f"⚠️ Skipping {folder_name} (not found)")
            continue
            
        print(f"\n🎬 Processing: {folder_name} -> {split_name}")
        videos = list(input_split_dir.glob("*.mp4"))
        print(f"   Found {len(videos)} videos")
        
        if not videos:
            print(f"   ⚠️ No videos found in {input_split_dir}")
            continue
        
        manifest_lines = []
        
        for vid in tqdm(videos, desc="Total Progress"):
            results = process_video(vid, OUTPUT_ROOT, fa)
            manifest_lines.extend(results)
            
        # Save filelist
        with open(FILELISTS_DIR / f"{split_name}.txt", "w") as f:
            for line in manifest_lines:
                f.write(line + "\n")
        
        print(f"   ✓ Saved {len(manifest_lines)} videos to {split_name}.txt")
    
    # 4. Create Archive (if requested)
    if not args.no_zip:
        create_archive(BASE)
    
    # Final GPU stats
    if device == 'cuda':
        print(f"\n📊 Final GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")
        print(f"   Peak Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    
    print("\n🎉 Preprocessing Complete!")

if __name__ == "__main__":
    main()