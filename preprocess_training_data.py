import sys
import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import torch
import shutil
import argparse

# --- PATH SETUP ---
def setup_wav2lip_import():
    # Look for Wav2Lip in current dir
    p = Path("Wav2Lip")
    if p.exists():
        sys.path.append(str(p.resolve()))
        try:
            from face_detection import FaceAlignment, LandmarksType
            return FaceAlignment, LandmarksType
        except ImportError:
            pass
    print("❌ Critical: Wav2Lip not found. Run setup_wav2lip.py first.")
    sys.exit(1)

def compute_affine_transform(landmarks, target_size=256):
    if landmarks is None or len(landmarks) == 0: return None
    try:
        canonical_points = np.float32([[70, 90], [186, 90], [128, 140]]) * (target_size / 256.0)
        src_points = np.float32([
            np.mean(landmarks[36:42], axis=0),
            np.mean(landmarks[42:48], axis=0),
            landmarks[30]
        ])
        M, _ = cv2.estimateAffinePartial2D(src_points, canonical_points)
        return M
    except: return None

def process_video(video_path, output_root, fa):
    vid_name = video_path.stem
    save_dir = output_root / vid_name
    if save_dir.exists(): shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Audio
    os.system(f'ffmpeg -y -v error -i "{video_path}" -ac 1 -vn -acodec pcm_s16le -ar 16000 "{save_dir}/audio.wav"')

    cap = cv2.VideoCapture(str(video_path))
    batch, frames = [], []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()

    if not frames: return []

    # Process in batches
    valid_frames = 0
    batch_size = 8
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
        try:
            preds = fa.get_detections_for_batch(np.array(batch))
        except: continue

        for j, landmarks in enumerate(preds):
            if landmarks is None: continue
            M = compute_affine_transform(landmarks)
            if M is not None:
                warped = cv2.warpAffine(batch[j], M, (256, 256), flags=cv2.INTER_CUBIC)
                idx = i + j
                cv2.imwrite(str(save_dir / f"{idx:05d}.jpg"), warped)
                np.save(str(save_dir / f"{idx:05d}.npy"), M)
                valid_frames += 1

    return [vid_name] if valid_frames > 5 else []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--output_root', type=str, required=True)
    parser.add_argument('--filelist_root', type=str, required=True)
    parser.add_argument('--no-zip', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Preprocessing on {device}")
    
    FaceAlignment, LandmarksType = setup_wav2lip_import()
    fa = FaceAlignment(LandmarksType._2D, flip_input=False, device=device)

    data_root = Path(args.data_root)
    for split in ['train', 'val', 'test']:
        split_dir = data_root / split
        if not split_dir.exists() and split == 'val': split_dir = data_root / 'valid'
        
        if not split_dir.exists():
            print(f"⚠️ Skipping {split}: Not found")
            continue

        print(f"🎬 Processing {split}...")
        videos = list(split_dir.glob("*.mp4"))
        manifest = []
        for v in tqdm(videos):
            manifest.extend(process_video(v, Path(args.output_root), fa))

        os.makedirs(args.filelist_root, exist_ok=True)
        with open(Path(args.filelist_root) / f"{split}.txt", "w") as f:
            f.write("\n".join(manifest))

if __name__ == "__main__":
    main()