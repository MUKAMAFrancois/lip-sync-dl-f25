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
import math

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
            print("FaceAlignment loaded for LMD.")
        except Exception as e:
            print(f"Could not load FaceAlignment: {e}")
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

    def calculate_psnr(self, gt_img, gen_img):
        """Peak Signal-to-Noise Ratio (Higher is better)"""
        mse = np.mean((gt_img - gen_img) ** 2)
        if mse == 0:
            return 100.0
        return 20 * math.log10(255.0 / math.sqrt(mse))

    def calculate_vlap(self, img):
        """Variance of Laplacian - measure of sharpness (Higher is sharper)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def run_fid(self, gt_path, gen_path):
        """Fréchet Inception Distance (Lower is better)"""
        print("Calculating FID...")
        try:
            # Calls pytorch-fid command line tool
            cmd = [sys.executable, "-m", "pytorch_fid", str(gt_path), str(gen_path), "--device", self.device]
            result = subprocess.run(cmd, capture_output=True, text=True)
            output = result.stdout.strip()
            
            # Extract number from "FID:  12.345"
            if "FID:" in output:
                val = float(output.split('FID:')[-1].strip())
                print(f"FID calculated: {val}")
                return val
            else:
                print(f"FID Output not recognized: {output}")
                return 0.0
        except Exception as e:
            print(f"FID Failed: {e}")
            return 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', type=str, required=True)
    parser.add_argument('--gen_path', type=str, required=True)
    args = parser.parse_args()

    # Validate Paths
    gt_root = Path(args.gt_path)
    gen_root = Path(args.gen_path)
    
    if not gt_root.exists():
        print(f"GT Path not found: {gt_root}")
        sys.exit(1)
        
    if not gen_root.exists():
        print(f"Gen Path not found: {gen_root}")
        print(" (Did inference fail? Folder is missing.)")
        sys.exit(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Starting Evaluation on {device}...")
    evaluator = Evaluator(device)
    
    video_folders = [x for x in gen_root.iterdir() if x.is_dir()]
    
    all_ssim = []
    all_lmd = []
    all_psnr = []
    all_vlap_gen = []
    all_vlap_gt = []
    
    print(f"Processing {len(video_folders)} videos...")
    
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

            # SSIM
            all_ssim.append(evaluator.calculate_ssim(gt_img, gen_img))
            
            # PSNR
            all_psnr.append(evaluator.calculate_psnr(gt_img, gen_img))

            # VLAP (Sharpness)
            all_vlap_gen.append(evaluator.calculate_vlap(gen_img))
            all_vlap_gt.append(evaluator.calculate_vlap(gt_img))
            
            # LMD (Sample every 10th frame for speed)
            if int(gen_f.stem) % 10 == 0: 
                lmd = evaluator.calculate_lmd(gt_img, gen_img)
                if lmd is not None: all_lmd.append(lmd)

    # Calculate FID over the whole set
    fid_score = evaluator.run_fid(gt_root, gen_root)

    print("\n" + "="*40)
    print("FINAL METRICS REPORT")
    print("="*40)
    print(f"SSIM (Visual Quality):  {np.mean(all_ssim) if all_ssim else 0.0:.4f} (Target: > 0.8)")
    print(f"PSNR (Reconstruction):  {np.mean(all_psnr) if all_psnr else 0.0:.4f} (Target: > 30.0)")
    print(f"VLAP (Sharpness Gen):   {np.mean(all_vlap_gen) if all_vlap_gen else 0.0:.4f} (Higher is sharper)")
    print(f"VLAP (Sharpness GT):    {np.mean(all_vlap_gt) if all_vlap_gt else 0.0:.4f}")
    print(f"LMD (Lip Sync Error):   {np.mean(all_lmd) if all_lmd else 0.0:.4f} (Target: < 3.5)")
    print(f"FID (Realism):          {fid_score:.4f} (Lower is better)")
    print("="*40)

    # Save to file
    with open("evaluation_results.txt", "w") as f:
        f.write(f"SSIM: {np.mean(all_ssim) if all_ssim else 0.0}\n")
        f.write(f"PSNR: {np.mean(all_psnr) if all_psnr else 0.0}\n")
        f.write(f"VLAP_Gen: {np.mean(all_vlap_gen) if all_vlap_gen else 0.0}\n")
        f.write(f"LMD: {np.mean(all_lmd) if all_lmd else 0.0}\n")
        f.write(f"FID: {fid_score}\n")

if __name__ == "__main__":
    main()