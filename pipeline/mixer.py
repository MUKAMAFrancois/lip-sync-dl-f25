import subprocess
from pathlib import Path

class AudioMixer:
    def __init__(self):
        pass

    def mix_audio(self, speech_track, background_track, output_path, bg_volume=1.0):
        """
        Mixes two audio tracks together into one file.
        
        Args:
            speech_track: The clean Aligned English TTS (Foreground).
            background_track: The 'no_vocals' track from Demucs (Background).
            output_path: Where to save the final mixed audio.
            bg_volume: Volume multiplier for background (1.0 = original).
                       Sometimes you want to lower background slightly (e.g., 0.8) so speech is clearer.
        """
        print(f"🎛️ Mixing Audio Layers...")
        
        # FFMPEG Complex Filter:
        # [0:a] is input 0 (speech)
        # [1:a] is input 1 (background) -> we apply volume change here if needed
        # amix=inputs=2:duration=first -> Mix 2 inputs, stop when the FIRST input ends (usually the video duration)
        
        # Note: 'duration=first' ensures the file matches the aligned speech track length (which matches video).
        
        cmd = f'ffmpeg -y -v error -i "{speech_track}" -i "{background_track}" ' \
              f'-filter_complex "[1:a]volume={bg_volume}[bg];[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2[a]" ' \
              f'-map "[a]" "{output_path}"'
              
        try:
            subprocess.run(cmd, shell=True, check=True)
            print(f"✅ Mixed Audio Saved: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"❌ Mixing Failed: {e}")
            raise

if __name__ == "__main__":
    # Test Mixing
    mixer = AudioMixer()
    mixer.mix_audio("aligned_speech.wav", "accompaniment.wav", "final_mix_test.wav")