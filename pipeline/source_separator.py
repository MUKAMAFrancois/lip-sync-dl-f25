# pip install demucs
import subprocess
import shutil
from pathlib import Path
import os

class SourceSeparator:
    def __init__(self):
        pass

    def separate(self, audio_path, output_dir):
        """
        Separates audio into 'vocals' and 'no_vocals' (accompaniment) using Demucs.
        
        Args:
            audio_path: Path to the input audio/video file.
            output_dir: Where to save the separated tracks.
            
        Returns:
            dict: Paths to {'vocals': Path, 'accompaniment': Path}
        """
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        print(f"🎸 Separating Background from: {audio_path.name}")
        
        # Demucs Command Structure:
        # -n htdemucs: Use the Hybrid Transformer model (Fast & SOTA)
        # --two-stems=vocals: Only split into "vocals" and "everything else" (saves time)
        # -o {output_dir}: Output directory
        cmd = [
            "demucs",
            "-n", "htdemucs",
            "--two-stems=vocals",
            "-o", str(output_dir),
            str(audio_path)
        ]
        
        try:
            # Run Demucs
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Demucs creates a subfolder structure: output_dir/htdemucs/{filename}/...
            # We need to find the files and verify they exist.
            model_name = "htdemucs"
            track_name = audio_path.stem
            separated_folder = output_dir / model_name / track_name
            
            vocals_path = separated_folder / "vocals.wav"
            accompaniment_path = separated_folder / "no_vocals.wav"
            
            if not vocals_path.exists() or not accompaniment_path.exists():
                raise FileNotFoundError("Demucs output files not found.")
                
            print(f"✅ Separation Complete.")
            print(f"   🎤 Vocals: {vocals_path.name}")
            print(f"   🎹 Accompaniment: {accompaniment_path.name}")
            
            return {
                "vocals": vocals_path,
                "accompaniment": accompaniment_path
            }
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Demucs Separation Failed: {e}")
            raise
        except Exception as e:
            print(f"❌ Error in Source Separation: {e}")
            raise

if __name__ == "__main__":
    # Test
    # Use the extracted audio from your main pipeline run
    TEST_AUDIO = Path("data/mtedx/video/de/train/dubbing_output/original_audio_extracted.wav")
    if TEST_AUDIO.exists():
        sep = SourceSeparator()
        sep.separate(TEST_AUDIO, "test_separation_output")