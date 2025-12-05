# pipeline/main.py (Updated)
import os
import argparse
from pathlib import Path
import shutil
import subprocess

# Import modules
from asr import ASRProcessor
from mt import MTProcessor
from tts import TTSProcessor
from duration_aligner import DurationAligner
from source_separator import SourceSeparator  
from mixer import AudioMixer                  

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
    video_path = Path(video_path)
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
    # This might take 30-60s on CPU
    separated_tracks = separator.separate(ref_audio_path, output_dir / "demucs")
    accompaniment_path = separated_tracks['accompaniment']
    
    # --- Step 2: ASR ---
    print("\n--- Step 2: ASR (Whisper) ---")
    asr = ASRProcessor()
    # Use the ORIGINAL audio for transcription (contains vocals)
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
        # Use original audio as speaker reference
        tts.generate_audio(seg['text'], str(ref_audio_path), str(clip_path), language="en")

    # --- Step 5: Duration Alignment ---
    print("\n--- Step 5: Duration Alignment ---")
    aligner = DurationAligner()
    clean_speech_track = output_dir / "aligned_speech_clean.wav"
    aligner.align_and_merge(str(video_path), tts_clips_dir, segments_path, clean_speech_track)

    # --- INTERMEDIATE STEP: Create Final Audio Mix (Validation) ---
    # We mix clean speech + background NOW so we can verify the result audibly
    print("\n--- Mixing Final Audio (For Validation) ---")
    mixer = AudioMixer()
    final_dubbed_audio = output_dir / "final_dubbed_audio.wav"
    mixer.mix_audio(clean_speech_track, accompaniment_path, final_dubbed_audio, bg_volume=0.8)

    # --- Step 6: Wav2Lip (Placeholder) ---
    print("\n--- Step 6: Wav2Lip (Next Phase) ---")
    print("To complete the video, you will run Wav2Lip with:")
    print(f"  --face {video_path}")
    print(f"  --audio {clean_speech_track}")
    
    print("\nPipeline Complete!")
    print(f"🎧 Listen to this file to verify the Dub: {final_dubbed_audio}")
    return final_dubbed_audio

if __name__ == "__main__":
    # Point to your test video
    BASE_DIR = Path(__file__).parent.parent
    TEST_VIDEO = BASE_DIR / "data/mtedx/video/de/train/0JI-oFgsdXw.mp4" 
    if TEST_VIDEO.exists():
        run_dubbing_pipeline(TEST_VIDEO, source_lang="german")