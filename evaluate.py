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
import sys

warnings.filterwarnings("ignore")

class Evaluator:
    def __init__(self, device='cuda'):
        self.device = device
        try:
            # FIX: Handle API change in face_alignment
            try:
                l_type = face_alignment.LandmarksType._2D
            except AttributeError:
                l_type = face_alignment.LandmarksType.TWO_D
                
            self.fa = face_alignment.FaceAlignment(l_type, device=device)
            print("✅ FaceAlignment loaded for LMD.")
        except Exception as e:
            print(f"⚠️ Could not load FaceAlignment: {e}")
            self.fa = None

    def calculate_lmd(self, gt_img, gen_img):
        if self.fa is None: return None
        try:
            gt_rgb = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
            gen_rgb = cv2.cvtColor(gen_img, cv2.COLOR_BGR2RGB)
            
            pts_gt = self.fa.get_landmarks_from_image(gt_rgb)
            pts_gen = self.fa.get_landmarks_from_image(gen_rgb)
            
            if pts_gt is None or pts_gen is None: return None
            
            # Unpack if list
            if isinstance(pts_gt, list): pts_gt = pts_gt[0]
            if isinstance(pts_gen, list): pts_gen = pts_gen[0]
            
            # Mouth Hull indices: 48-68
            mouth_gt = pts_gt[48:68]
            mouth_gen = pts_gen[48:68]
            
            return np.mean(np.linalg.norm(mouth_gt - mouth_gen, axis=1))
        except Exception:
            return None

    def calculate_ssim(self, gt_img, gen_img):
        gt_gray = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
        gen_gray = cv2.cvtColor(gen_img, cv2.COLOR_BGR2GRAY)
        return ssim(gt_gray, gen_gray)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', type=str, required=True)
    parser.add_argument('--gen_path', type=str, required=True)
    args = parser.parse_args()

    # Validate Paths
    gt_root = Path(args.gt_path)
    gen_root = Path(args.gen_path)
    
    if not gt_root.exists():
        print(f"❌ GT Path not found: {gt_root}")
        sys.exit(1)
        
    if not gen_root.exists():
        print(f"❌ Gen Path not found: {gen_root}")
        # Create it just to prevent crash if empty, but warn
        print("   (Did inference fail? Folder is missing.)")
        sys.exit(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Starting Evaluation on {device}...")
    evaluator = Evaluator(device)
    
    video_folders = [x for x in gen_root.iterdir() if x.is_dir()]
    
    all_ssim = []
    all_lmd = []
    
    print(f"📊 Processing {len(video_folders)} videos...")
    
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
            
            # Resize GT to match Gen if needed
            if gen_img.shape != gt_img.shape:
                gt_img = cv2.resize(gt_img, (gen_img.shape[1], gen_img.shape[0]))

            all_ssim.append(evaluator.calculate_ssim(gt_img, gen_img))
            
            if int(gen_f.stem) % 10 == 0: # Check every 10th frame
                lmd = evaluator.calculate_lmd(gt_img, gen_img)
                if lmd is not None: all_lmd.append(lmd)

    print("\n" + "="*40)
    print(f"SSIM: {np.mean(all_ssim) if all_ssim else 0.0:.4f}")
    print(f"LMD: {np.mean(all_lmd) if all_lmd else 0.0:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()