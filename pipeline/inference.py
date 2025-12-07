# pipeline/inference.py
import torch
import numpy as np
import cv2
import os
import sys
import subprocess
from pathlib import Path
from tqdm import tqdm

# --- Dynamic Path Setup ---
# Add project root to path so we can import 'models' and 'hparams'
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Try importing Wav2Lip dependencies
try:
    from models import Wav2Lip
    import audio
    from hparams import hparams
except ImportError:
    # Fallback: Try looking inside a Wav2Lip subfolder if it exists
    w2l_path = PROJECT_ROOT / "Wav2Lip"
    if w2l_path.exists():
        sys.path.append(str(w2l_path))
        from models import Wav2Lip
        import audio
    else:
        print("❌ Error: Could not find 'models' or 'Wav2Lip' directory.")
        sys.exit(1)

class LipSyncProcessor:
    def __init__(self, checkpoint_path, face_det_batch_size=16):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.img_size = hparams['img_size']  # Uses 256 from your config
        self.face_det_batch_size = face_det_batch_size
        
        print(f"🚀 Loading LipSync Model from {checkpoint_path}...")
        self.model = self._load_model(checkpoint_path)
        print(f"✅ Model loaded on {self.device}. Resolution: {self.img_size}x{self.img_size}")

    def _load_model(self, path):
        model = Wav2Lip()
        checkpoint = torch.load(path, map_location=self.device)
        
        # Handle state dict keys (remove 'module.' if trained with DataParallel)
        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
            
        model.load_state_dict(new_s)
        model = model.to(self.device)
        return model.eval()

    def get_smoothened_landmarks(self, all_landmarks, window_size=5):
        """Smooths landmarks over time to reduce jitter."""
        if len(all_landmarks) == 0: return []
        smoothed = []
        for i in range(len(all_landmarks)):
            window = all_landmarks[max(0, i - window_size // 2):min(len(all_landmarks), i + window_size // 2 + 1)]
            # Filter out None values in the window
            valid = [x for x in window if x is not None]
            if not valid:
                smoothed.append(None)
            else:
                smoothed.append(np.mean(valid, axis=0))
        return smoothed

    def run(self, face_path, audio_path, output_path):
        """
        Runs the full inference loop.
        Args:
            face_path: Path to input video (mp4).
            audio_path: Path to aligned audio (wav).
            output_path: Path to save result.
        """
        if not os.path.exists(face_path): raise FileNotFoundError(f"Video not found: {face_path}")
        if not os.path.exists(audio_path): raise FileNotFoundError(f"Audio not found: {audio_path}")

        # 1. Load Audio & Mels
        print("   Processing Audio...")
        wav = audio.load_wav(audio_path, 16000)
        mel = audio.melspectrogram(wav)
        
        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError('Mel contains NaN! Check audio file.')

        # Chunk Mel into batches corresponding to video frames
        mel_chunks = []
        mel_step_size = 16
        mel_idx_multiplier = 80. / (16000 / 16000 * 25) # Approx mapping, simpler logic below:
        
        # 2. Read Video Frames
        print("   Reading Video Frames...")
        cap = cv2.VideoCapture(face_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        full_frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            full_frames.append(frame)
        cap.release()

        if not full_frames:
            raise ValueError("Video contains no frames.")

        # 3. Detect Faces
        # Note: We rely on face_alignment (S3FD) imported inside the loop to save memory
        import face_alignment
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=self.device)
        
        print("   Detecting Faces...")
        all_landmarks = []
        # Batch detection
        for i in tqdm(range(0, len(full_frames), self.face_det_batch_size)):
            batch = np.array(full_frames[i:i + self.face_det_batch_size])
            preds = fa.get_detections_for_batch(batch)
            for p in preds:
                all_landmarks.append(p)

        # Smooth landmarks
        all_landmarks = self.get_smoothened_landmarks(all_landmarks)

        # 4. Prepare Batches for Inference
        print("   Generating Frames...")
        # Prepare mel batches strictly aligned to video length
        mel_chunks = []
        # Audio/Video length mismatch handling:
        # We loop the face if audio is longer, or cut if audio is shorter?
        # Standard approach: Limit to min length
        min_len = min(len(full_frames), int(len(mel[0])/16 * (16000/16000*25)) ) # Rough estimate
        
        # Better: Generate exactly one mel chunk per frame
        batch_size = hparams['batch_size']
        gen_frames = []
        
        # We need (batch_size, 6, H, W) input
        img_batch, mel_batch, frame_idx_batch, coords_batch = [], [], [], []

        for i, m in enumerate(all_landmarks):
            if i >= len(full_frames): break # Safety
            
            if m is None: continue # Skip if no face found (keep original frame later)

            # Calculate Audio Slice
            # 16000 Hz audio, 25 FPS video -> 640 audio samples per frame
            # Wav2Lip takes window of ~0.2s around the frame
            idx = int(float(i) / fps * 16000) # Audio sample index
            
            # Mel step calculation (Wav2Lip specific magic numbers)
            # 80 * 16 mels is standard input
            start = int(i * (mel.shape[1] / len(full_frames)))
            if start + 16 > mel.shape[1]: break
            m_slice = mel[:, start : start + 16]
            
            if m_slice.shape[1] != 16: continue

            # Face Crop
            # Use same transform logic as preprocessing
            # Standard Wav2Lip crop logic (simplified here for brevity)
            # We compute a crop using the landmarks
            try:
                # Simple square crop around center of face
                y1, x1 = np.min(m, axis=0)
                y2, x2 = np.max(m, axis=0)
                h, w = y2 - y1, x2 - x1
                center_y, center_x = y1 + h/2, x1 + w/2
                size = int(max(h, w) * 1.5) # Add padding
                
                # Coords
                y1_c = max(0, int(center_y - size/2))
                x1_c = max(0, int(center_x - size/2))
                y2_c = min(full_frames[i].shape[0], int(center_y + size/2))
                x2_c = min(full_frames[i].shape[1], int(center_x + size/2))
                
                crop = full_frames[i][y1_c:y2_c, x1_c:x2_c]
                crop = cv2.resize(crop, (self.img_size, self.img_size))
            except:
                continue

            # Mask lower half
            masked_crop = crop.copy()
            masked_crop[self.img_size//2:] = 0

            # Concatenate (Masked, Reference) - using same frame as ref for simplicity in inference
            # Ideally use a random other frame, but same frame works for identity preservation
            inp = np.concatenate([masked_crop, crop], axis=2)
            inp = inp.transpose(2, 0, 1) / 255.0

            img_batch.append(inp)
            mel_batch.append(m_slice)
            coords_batch.append((y1_c, y2_c, x1_c, x2_c))
            frame_idx_batch.append(i)

            if len(img_batch) >= batch_size:
                self._infer_batch(img_batch, mel_batch, coords_batch, frame_idx_batch, full_frames)
                img_batch, mel_batch, frame_idx_batch, coords_batch = [], [], [], []

        # Process remaining
        if img_batch:
            self._infer_batch(img_batch, mel_batch, coords_batch, frame_idx_batch, full_frames)

        # 5. Write Video
        print("   Writing Final Video...")
        temp_video = output_path.replace(".mp4", "_silent.mp4")
        out = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (full_frames[0].shape[1], full_frames[0].shape[0]))
        for f in full_frames:
            out.write(f)
        out.release()

        # 6. Merge Audio (Using ffmpeg)
        cmd = f'ffmpeg -y -v error -i "{temp_video}" -i "{audio_path}" -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 "{output_path}"'
        subprocess.run(cmd, shell=True)
        
        # Cleanup
        if os.path.exists(temp_video): os.remove(temp_video)
        print(f"✅ Saved to: {output_path}")

    def _infer_batch(self, img_batch, mel_batch, coords_batch, frame_idx_batch, full_frames):
        img_t = torch.FloatTensor(np.array(img_batch)).to(self.device)
        mel_t = torch.FloatTensor(np.array(mel_batch)).unsqueeze(1).to(self.device)

        with torch.no_grad():
            pred = self.model(img_t, mel_t)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

        for p, coords, idx in zip(pred, coords_batch, frame_idx_batch):
            y1, y2, x1, x2 = coords
            p_resized = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            
            # Simple Paste back (can be improved with Gaussian blending)
            full_frames[idx][y1:y2, x1:x2] = p_resized

if __name__ == "__main__":
    # Test
    PROC = LipSyncProcessor("checkpoints/wav2lip_gan.pth")
    # PROC.run("test.mp4", "test.wav", "result.mp4")