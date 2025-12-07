# pipeline/main.py
import os
import argparse
from pathlib import Path
import subprocess

# Import modules
from asr import ASRProcessor
from mt import MTProcessor
from tts import TTSProcessor
from duration_aligner import DurationAligner
from source_separator import SourceSeparator  
from mixer import AudioMixer                  
from inference import LipSyncProcessor  # <--- IMPORTING THE NEW MODULE

def extract_audio_from_video(video_path, output_wav_path):
    if output_wav_path.exists():
        return True
    print(f"   Extracting audio -> {output_wav_path.name}")
    try:
        cmd = f'ffmpeg -y -v error -i "{video_path}" -vn -acodec pcm_s16le -ar 44100 -ac 2 "{output_wav_path}"'
        subprocess.run(cmd, shell=True, check=True)
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def run_dubbing_pipeline(video_path, source_lang, target_lang="english"):
    video_path = Path(video_path).resolve()
    output_dir = video_path.parent / "dubbing_output"
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n🎥 Starting Dubbing Pipeline for: {video_path.name}")

    # --- Step 0: Extract Audio ---
    ref_audio_path = output_dir / "original_audio.wav"
    if not extract_audio_from_video(video_path, ref_audio_path):
        return

    # --- Step 1: Source Separation (Demucs) ---
    print("\n--- Step 1: Source Separation (Demucs) ---")
    separator = SourceSeparator()
    separated_tracks = separator.separate(ref_audio_path, output_dir / "demucs")
    accompaniment_path = separated_tracks['accompaniment']
    
    # --- Step 2: ASR ---
    print("\n--- Step 2: ASR (Whisper) ---")
    asr = ASRProcessor()
    asr_result = asr.transcribe(ref_audio_path, language=None)
    segments_path = output_dir / "segments.json"
    asr.save_segments(asr_result, segments_path)
    
    # --- Step 3: MT ---
    print("\n--- Step 3: MT (NLLB) ---")
    mt = MTProcessor()
    translated_segments = []
    for seg in asr_result['segments']:
        text = seg['text']
        trans_text = mt.translate(text, source_lang=source_lang, target_lang=target_lang)
        new_seg = seg.copy()
        new_seg['text'] = trans_text
        translated_segments.append(new_seg)

    # --- Step 4: TTS ---
    print("\n--- Step 4: TTS (XTTS-v2) ---")
    tts = TTSProcessor()
    tts_clips_dir = output_dir / "tts_clips"
    tts_clips_dir.mkdir(exist_ok=True)
    
    for i, seg in enumerate(translated_segments):
        clip_path = tts_clips_dir / f"segment_{i}.wav"
        tts.generate_audio(seg['text'], str(ref_audio_path), str(clip_path), language="en")

    # --- Step 5: Duration Alignment ---
    print("\n--- Step 5: Duration Alignment ---")
    aligner = DurationAligner()
    clean_speech_track = output_dir / "aligned_speech_clean.wav"
    aligner.align_and_merge(str(video_path), tts_clips_dir, segments_path, clean_speech_track)

    # --- Intermediate Mix (Clean Speech + BG Music) ---
    print("\n--- Mixing Final Audio (For Sync) ---")
    mixer = AudioMixer()
    final_dubbed_audio = output_dir / "final_dubbed_audio.wav"
    mixer.mix_audio(clean_speech_track, accompaniment_path, final_dubbed_audio, bg_volume=0.8)

    # --- Step 6: Visual Dubbing (Wav2Lip) ---
    print("\n--- Step 6: Visual Dubbing (Inference) ---")
    
    # Path to your latest checkpoint (Ensure you point to the one trained in train.py)
    # If no custom training happened yet, use the base pretrained GAN
    checkpoint_path = Path("checkpoints/wav2lip_gan.pth") 
    
    # If a trained checkpoint exists in checkpoints/, prioritize it
    # custom_ckpt = Path("checkpoints/checkpoint_step000000001.pth") # Example name
    # if custom_ckpt.exists(): checkpoint_path = custom_ckpt

    lip_syncer = LipSyncProcessor(checkpoint_path=str(checkpoint_path))
    
    final_video_path = output_dir / f"{video_path.stem}_dubbed_en.mp4"
    
    lip_syncer.run(
        face_path=str(video_path), 
        audio_path=str(final_dubbed_audio), 
        output_path=str(final_video_path)
    )
    
    print("\n🎉 Pipeline Complete!")
    print(f"🎬 Final Video: {final_video_path}")
    return final_video_path

if __name__ == "__main__":
    # Point to your test video
    BASE_DIR = Path(__file__).resolve().parent.parent
    # Example video path
    TEST_VIDEO = BASE_DIR / "data/german/test/sample.mp4" 
    
    if TEST_VIDEO.exists():
        run_dubbing_pipeline(TEST_VIDEO, source_lang="german")
    else:
        print(f"Please check TEST_VIDEO path in main.py: {TEST_VIDEO}")