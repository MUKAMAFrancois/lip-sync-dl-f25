# pip install openai-whisper
import whisper
import torch
import os
import json
from pathlib import Path

class ASRProcessor:
    # Class-level variable to store the model (Singleton)
    _model_cache = None
    _model_size_cache = None

    def __init__(self, model_size="medium"):
        """
        Initializes the Whisper ASR model.
        Uses a Singleton pattern to avoid reloading if already in memory.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Check if we need to load (or reload) the model
        if ASRProcessor._model_cache is None or ASRProcessor._model_size_cache != model_size:
            print(f"...Loading Whisper ({model_size}) on {self.device}...")
            try:
                # Load to the class variable, not just self
                ASRProcessor._model_cache = whisper.load_model(model_size, device=self.device)
                ASRProcessor._model_size_cache = model_size
                print("Good: Whisper loaded successfully.")
            except Exception as e:
                print(f"!!! Error loading Whisper: {e}")
                raise
        else:
            print(f"Good: Using cached Whisper model ({model_size}).")

        # Point self.model to the shared class variable
        self.model = ASRProcessor._model_cache

    def transcribe(self, audio_path, language=None):
        """
        Transcribes audio file to text with timestamps.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(f"Transcribing: {audio_path}...")
        
        options = {}
        if language:
            options["language"] = language

        # Run transcription
        result = self.model.transcribe(str(audio_path), **options)
        
        detected_lang = result.get('language', 'unknown')
        print(f"Good. Transcription complete. Detected Language: {detected_lang}")
        
        return result

    def save_segments(self, result, output_path):
        """Saves the segments to a JSON file for inspection."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result['segments'], f, indent=4, ensure_ascii=False)
        print(f"Good. Segments saved to: {output_path}")

# --- Testing Block ---
if __name__ == "__main__":
    
    # Update this path to your specific file
    TEST_AUDIO = Path("muavic-repo/data/muavic/de/audio/train/_Hk4MOw9gsA/_Hk4MOw9gsA_0000.wav")
    if TEST_AUDIO.exists():
        # First initialization (Will Load Model)
        print("\n--- 1st Call (Should Load) ---")
        asr1 = ASRProcessor(model_size="medium")
        
        # Second initialization (Should SKIP Loading)
        print("\n--- 2nd Call (Should Cache) ---")
        asr2 = ASRProcessor(model_size="medium")
        
        # Transcribe
        output = asr1.transcribe(TEST_AUDIO)
        
        print("\n--- Sample Output ---")
        print(f"Full Text: {output['text']}")
        if output['segments']:
            first_seg = output['segments'][0]
            print(f"First Segment: [{first_seg['start']:.2f}s -> {first_seg['end']:.2f}s] '{first_seg['text']}'")
            
        asr1.save_segments(output, "test_asr_output.json")
    else:
        print(f"!! Test file not found at {TEST_AUDIO}. Please verify path.")