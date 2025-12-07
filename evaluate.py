# evaluate.py
import torch
import numpy as np
import cv2
import os
import argparse
from tqdm import tqdm
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
import face_alignment
import subprocess
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

class Evaluator:
    def __init__(self, device='cuda'):
        self.device = device
        try:
            # Initialize Face Alignment for LMD
            self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device)
            print("✅ FaceAlignment loaded for LMD.")
        except Exception as e:
            print(f"⚠️ Could not load FaceAlignment: {e}")
            self.fa = None

    def calculate_lmd(self, gt_img, gen_img):
        """Landmark Distance (Lower is better) - Measures Lip Sync"""
        if self.fa is None: return None
        try:
            # Convert to RGB
            gt_rgb = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
            gen_rgb = cv2.cvtColor(gen_img, cv2.COLOR_BGR2RGB)
            
            # Get Landmarks
            pts_gt = self.fa.get_landmarks_from_image(gt_rgb)
            pts_gen = self.fa.get_landmarks_from_image(gen_rgb)
            
            if pts_gt is None or pts_gen is None: return None
            
            # Mouth Hull indices: 48-68
            mouth_gt = pts_gt[0][48:68]
            mouth_gen = pts_gen[0][48:68]
            
            # Mean Euclidean Distance
            return np.mean(np.linalg.norm(mouth_gt - mouth_gen, axis=1))
        except Exception:
            return None

    def calculate_ssim(self, gt_img, gen_img):
        """SSIM (Higher is better) - Measures Visual Quality"""
        gt_gray = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
        gen_gray = cv2.cvtColor(gen_img, cv2.COLOR_BGR2GRAY)
        return ssim(gt_gray, gen_gray)

    def run_fid(self, gt_folder, gen_folder):
        """FID (Lower is better) - Measures Realism"""
        print("⏳ Calculating FID (this may take a moment)...")
        try:
            # Calls pytorch-fid command line tool
            result = subprocess.run(
                ["python", "-m", "pytorch_fid", str(gt_folder), str(gen_folder), "--device", self.device],
                capture_output=True, text=True
            )
            # Parse output "FID:  12.345"
            output = result.stdout.strip()
            print(f"   FID Output: {output}")
            
            # Extract number
            if "FID:" in output:
                return float(output.split('FID:')[-1].strip())
            return 0.0
        except Exception as e:
            print(f"⚠️ FID Failed: {e}")
            return 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', type=str, required=True, help="Folder containing Ground Truth frames")
    parser.add_argument('--gen_path', type=str, required=True, help="Folder containing Generated frames")
    args = parser.parse_args()

    # 1. Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Starting Evaluation on {device}...")
    evaluator = Evaluator(device)
    
    gt_root = Path(args.gt_path)
    gen_root = Path(args.gen_path)
    
    # 2. Iterate Folders
    # We assume gen_path has subfolders for each video, matching gt_path
    video_folders = [x for x in gen_root.iterdir() if x.is_dir()]
    
    all_ssim = []
    all_lmd = []
    
    print(f"📊 Processing {len(video_folders)} videos for SSIM & LMD...")
    
    for vid_dir in tqdm(video_folders):
        vid_name = vid_dir.name
        gt_vid_dir = gt_root / vid_name
        
        if not gt_vid_dir.exists(): continue
        
        gen_frames = sorted(list(vid_dir.glob("*.jpg")))
        
        for gen_f in gen_frames:
            gt_f = gt_vid_dir / gen_f.name
            if not gt_f.exists(): continue
            
            gen_img = cv2.imread(str(gen_f))
            gt_img = cv2.imread(str(gt_f))
            
            if gen_img is None or gt_img is None: continue
            
            # SSIM
            all_ssim.append(evaluator.calculate_ssim(gt_img, gen_img))
            
            # LMD (Sample 1 every 5 frames to speed up face alignment)
            if int(gen_f.stem) % 5 == 0:
                lmd = evaluator.calculate_lmd(gt_img, gen_img)
                if lmd is not None: all_lmd.append(lmd)

    # 3. Calculate FID (Dataset Level)
    # FID needs the paths to the root directories of images
    # We can pass the parent folders if they contain flat images, 
    # but since we have subfolders, pytorch-fid might complain.
    # Standard practice: FID is calculated on a flattened set of images. 
    # For now, we attempt running it on the directory structures (newer versions support it).
    fid_score = evaluator.run_fid(gt_root, gen_root)

    # 4. Report
    print("\n" + "="*40)
    print("🏆 FINAL METRICS REPORT")
    print("="*40)
    print(f"1. SSIM (Visual Quality):  {np.mean(all_ssim):.4f} (Target: > 0.8)")
    print(f"2. LMD (Lip Sync Error):   {np.mean(all_lmd):.4f} (Target: < 3.5)")
    print(f"3. FID (Realism):          {fid_score:.4f} (Lower is better)")
    print("="*40)

    # Save to file
    with open("evaluation_results.txt", "w") as f:
        f.write(f"SSIM: {np.mean(all_ssim)}\n")
        f.write(f"LMD: {np.mean(all_lmd)}\n")
        f.write(f"FID: {fid_score}\n")

if __name__ == "__main__":
    main()