# pipeline/main.py
import os
import argparse
from pathlib import Path
import subprocess
import traceback
from tqdm import tqdm

# Import modules
# Ensure these files exist in your pipeline/ folder
try:
    from asr import ASRProcessor
    from mt import MTProcessor
    from tts import TTSProcessor
    from duration_aligner import DurationAligner
    from source_separator import SourceSeparator  
    from mixer import AudioMixer                  
    from inference import LipSyncProcessor
except ImportError:
    # Fix for running as a script where python path might not include current dir
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from asr import ASRProcessor
    from mt import MTProcessor
    from tts import TTSProcessor
    from duration_aligner import DurationAligner
    from source_separator import SourceSeparator  
    from mixer import AudioMixer                  
    from inference import LipSyncProcessor

def extract_audio_from_video(video_path, output_wav_path):
    if output_wav_path.exists():
        return True
    # print(f"   Extracting audio -> {output_wav_path.name}")
    try:
        cmd = f'ffmpeg -y -v error -i "{video_path}" -vn -acodec pcm_s16le -ar 44100 -ac 2 "{output_wav_path}"'
        subprocess.run(cmd, shell=True, check=True)
        return True
    except Exception as e:
        print(f"❌ Error extracting audio: {e}")
        return False

def run_dubbing_pipeline(video_path, source_lang, target_lang="english", output_root=None):
    video_path = Path(video_path).resolve()
    
    # Define Output Directory
    # If output_root is provided, save there. Otherwise save next to video.
    if output_root:
        # Create a structure mirroring the input filename
        output_dir = Path(output_root) / video_path.stem
    else:
        output_dir = video_path.parent / f"dubbing_output_{video_path.stem}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🎥 === Processing: {video_path.name} ===")

    # --- Step 0: Extract Audio ---
    ref_audio_path = output_dir / "original_audio.wav"
    if not extract_audio_from_video(video_path, ref_audio_path):
        return

    try:
        # --- Step 1: Source Separation (Demucs) ---
        print("   1. Source Separation...")
        separator = SourceSeparator()
        separated_tracks = separator.separate(ref_audio_path, output_dir / "demucs")
        accompaniment_path = separated_tracks['accompaniment']
        
        # --- Step 2: ASR ---
        print("   2. ASR (Whisper)...")
        asr = ASRProcessor()
        asr_result = asr.transcribe(ref_audio_path, language=None)
        segments_path = output_dir / "segments.json"
        asr.save_segments(asr_result, segments_path)
        
        # --- Step 3: MT ---
        print("   3. Translation (NLLB)...")
        mt = MTProcessor()
        translated_segments = []
        for seg in asr_result['segments']:
            text = seg['text']
            trans_text = mt.translate(text, source_lang=source_lang, target_lang=target_lang)
            new_seg = seg.copy()
            new_seg['text'] = trans_text
            translated_segments.append(new_seg)

        # --- Step 4: TTS ---
        print("   4. TTS (XTTS-v2)...")
        tts = TTSProcessor()
        tts_clips_dir = output_dir / "tts_clips"
        tts_clips_dir.mkdir(exist_ok=True)
        
        for i, seg in enumerate(translated_segments):
            clip_path = tts_clips_dir / f"segment_{i}.wav"
            # Optimization: Skip if exists
            if not clip_path.exists():
                tts.generate_audio(seg['text'], str(ref_audio_path), str(clip_path), language="en")

        # --- Step 5: Duration Alignment ---
        print("   5. Alignment...")
        aligner = DurationAligner()
        clean_speech_track = output_dir / "aligned_speech_clean.wav"
        aligner.align_and_merge(str(video_path), tts_clips_dir, segments_path, clean_speech_track)

        # --- Intermediate Mix ---
        print("   6. Audio Mixing...")
        mixer = AudioMixer()
        final_dubbed_audio = output_dir / "final_dubbed_audio.wav"
        mixer.mix_audio(clean_speech_track, accompaniment_path, final_dubbed_audio, bg_volume=0.8)

        # --- Step 6: Visual Dubbing (Wav2Lip) ---
        print("   7. Visual Dubbing (Wav2Lip)...")
        
        # Checkpoint Strategy:
        # 1. Look for fine-tuned expert in checkpoints/
        # 2. Fallback to generic GAN
        checkpoint_path = Path("checkpoints/wav2lip_gan.pth")
        
        # NOTE: If you have a fine-tuned model from training, update this name!
        # fine_tuned = Path("checkpoints/checkpoint_step000003000.pth") 
        # if fine_tuned.exists(): checkpoint_path = fine_tuned

        lip_syncer = LipSyncProcessor(checkpoint_path=str(checkpoint_path))
        
        final_video_path = output_dir / f"{video_path.stem}_dubbed_en.mp4"
        
        lip_syncer.run(
            face_path=str(video_path), 
            audio_path=str(final_dubbed_audio), 
            output_path=str(final_video_path)
        )
        
        print(f"✅ Success! Saved to: {final_video_path}")
        return final_video_path

    except Exception as e:
        print(f"❌ Failed processing {video_path.name}: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Dubbing Pipeline")
    parser.add_argument('--input_path', type=str, required=True, help="Path to video file OR directory of videos")
    parser.add_argument('--output_root', type=str, default=None, help="Optional: Central folder to save all outputs")
    parser.add_argument('--source_lang', type=str, default="german")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    
    if not input_path.exists():
        print(f"❌ Error: Input path not found: {input_path}")
        exit(1)

    # Collect videos
    videos_to_process = []
    if input_path.is_file():
        videos_to_process.append(input_path)
    elif input_path.is_dir():
        print(f"📂 Scanning directory: {input_path}")
        # Recursive glob for .mp4
        videos_to_process = list(input_path.rglob("*.mp4"))
    
    print(f"🚀 Found {len(videos_to_process)} videos to process.")
    
    # Batch Loop
    success_count = 0
    for video in tqdm(videos_to_process, desc="Dubbing Videos"):
        result = run_dubbing_pipeline(video, args.source_lang, output_root=args.output_root)
        if result:
            success_count += 1
            
    print(f"\n🎉 Batch Complete. Successfully dubbed {success_count}/{len(videos_to_process)} videos.")