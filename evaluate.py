# pip install scikit-image face-alignment pytorch-fid
import os
import torch
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import face_alignment
from tqdm import tqdm
from pathlib import Path

# Initialize Face Alignment for LMD (Landmark Distance)
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda' if torch.cuda.is_available() else 'cpu')

def calculate_lmd(ground_truth_path, generated_path):
    """
    Calculates Landmark Distance (LMD) - Measures Lip Sync Accuracy.
    Lower is better.
    """
    gt = cv2.imread(ground_truth_path)
    gen = cv2.imread(generated_path)
    
    try:
        pts_gt = fa.get_landmarks(gt)[0]
        pts_gen = fa.get_landmarks(gen)[0]
        
        # Focus on Mouth Landmarks (Indices 48-68)
        mouth_gt = pts_gt[48:68]
        mouth_gen = pts_gen[48:68]
        
        # Euclidean distance
        distance = np.mean(np.linalg.norm(mouth_gt - mouth_gen, axis=1))
        return distance
    except:
        return None

def calculate_ssim(ground_truth_path, generated_path):
    """
    Calculates Structural Similarity (SSIM) - Measures Image Quality.
    Higher (closer to 1.0) is better.
    """
    gt = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
    gen = cv2.imread(generated_path, cv2.IMREAD_GRAYSCALE)
    return ssim(gt, gen)

def evaluate_folder(gt_folder, gen_folder):
    lmd_scores = []
    ssim_scores = []
    
    gt_files = sorted(list(Path(gt_folder).glob("*.jpg")))
    
    print(f"📊 Evaluating {len(gt_files)} frames...")
    
    for gt_file in tqdm(gt_files):
        gen_file = Path(gen_folder) / gt_file.name
        if not gen_file.exists(): continue
        
        # SSIM
        score = calculate_ssim(str(gt_file), str(gen_file))
        ssim_scores.append(score)
        
        # LMD
        lmd = calculate_lmd(str(gt_file), str(gen_file))
        if lmd is not None:
            lmd_scores.append(lmd)
            
    print("\n--- 📝 FINAL REPORT ---")
    print(f"SSIM (Quality): {np.mean(ssim_scores):.4f} (Target: > 0.85)")
    print(f"LMD (Sync):     {np.mean(lmd_scores):.4f} (Target: < 3.0)")
    
    # Note: FID is calculated separately using 'python -m pytorch_fid path/to/gt path/to/gen'
    print("To calculate FID (Realism), run:")
    print(f"python -m pytorch_fid {gt_folder} {gen_folder}")

if __name__ == "__main__":
    # Example Usage
    evaluate_folder("data/german/test/frames_gt", "data/german/test/frames_gen")