import os
import json
from pathlib import Path
from pydub import AudioSegment

class DurationAligner:
    def __init__(self):
        pass

    def speed_change(self, sound, speed=1.0):
        """
        Manually changes the speed of audio without changing pitch.
        """
        # Avoid extreme frame rates that crash Wav2Lip (keep between 8k and 48k)
        new_rate = int(sound.frame_rate * speed)
        # Safe clamp
        new_rate = max(8000, min(new_rate, 48000))
        
        sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={
            "frame_rate": new_rate
        })
        return sound_with_altered_frame_rate.set_frame_rate(sound.frame_rate)

    def align_and_merge(self, video_path, tts_clips_dir, segments_json_path, output_path):
        """
        Creates the 'Canvas' with Smart Positioning.
        """
        # 1. Get Total Duration
        try:
            original_track = AudioSegment.from_file(str(video_path))
            total_duration_ms = len(original_track)
        except:
            print("!! Could not read duration from video. Using JSON end time.")
            with open(segments_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                total_duration_ms = int(data[-1]['end'] * 1000) + 2000

        # 2. Create The Silent Canvas
        combined_audio = AudioSegment.silent(duration=total_duration_ms)
        
        # 3. Load Segments
        with open(segments_json_path, 'r', encoding='utf-8') as f:
            segments = json.load(f)

        print(f"... Aligning {len(segments)} segments with Smart Positioning...")

        for i, seg in enumerate(segments):
            tts_filename = f"segment_{i}.wav"
            tts_path = Path(tts_clips_dir) / tts_filename
            
            if not tts_path.exists():
                continue

            tts_audio = AudioSegment.from_file(tts_path)
            
            # Times in MS
            start_ms = int(seg['start'] * 1000)
            end_ms = int(seg['end'] * 1000)
            target_slot_ms = end_ms - start_ms
            current_tts_ms = len(tts_audio)

            # Ratio Calculation
            ratio = current_tts_ms / target_slot_ms
            
            final_clip = tts_audio

            # --- A. SPEED ADJUSTMENT ---
            if ratio > 1.1:
                # TTS is too long -> Speed Up
                speed_factor = min(ratio, 1.25)
                print(f"   Seg {i}: Speeding up {speed_factor:.2f}x")
                final_clip = self.speed_change(tts_audio, speed_factor)
                current_tts_ms = len(final_clip) # Update length after shrink
            
            # --- B. SMART POSITIONING (The Fix) ---
            # Calculate how much 'slack' (silence) is in this slot
            slack_ms = target_slot_ms - current_tts_ms
            
            if slack_ms > 0:
                if i == 0:
                    # INTRO CASE: If it's the first segment, align to the END.
                    # This fixes the "Speaking over Applause" bug.
                    # The speaker usually starts after the clapping.
                    placement_ms = start_ms + slack_ms - 200 # 200ms buffer from very end
                    print(f"   Seg {0}: Intro detected. Right-aligning speech to {placement_ms}ms")
                else:
                    # NORMAL CASE: Center the speech in the slot.
                    # This feels more natural than left-aligning.
                    placement_ms = start_ms + (slack_ms // 2)
            else:
                # No slack (or we sped it up to fit exactly), put at start
                placement_ms = start_ms

            # Safety check: ensure we don't place before 0
            placement_ms = max(0, placement_ms)

            # Overlay
            combined_audio = combined_audio.overlay(final_clip, position=placement_ms)

        # 4. Export
        combined_audio.export(output_path, format="wav")
        print(f"Okay: Aligned Audio saved to: {output_path}")
        return output_path

# --- Testing Block ---
if __name__ == "__main__":
    # To test this, we need to mock the inputs since we don't have the full loop yet.
    # But this script is ready to be imported into main.py
    print("Duration Aligner Module Ready.")