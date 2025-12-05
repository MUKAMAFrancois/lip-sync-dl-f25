import torch
import os
import sys
import re
from pydub import AudioSegment

# 1. Set Environment Variable to agree to Coqui License
os.environ["COQUI_TOS_AGREED"] = "1"

from TTS.api import TTS

class TTSProcessor:
    _model_cache = None

    def __init__(self, model_name="tts_models/multilingual/multi-dataset/xtts_v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if TTSProcessor._model_cache is None:
            print(f"🚀 Loading TTS Model ({model_name}) on {self.device}...")
            try:
                tts = TTS(model_name)
                if self.device == "cuda":
                    tts.to(self.device)
                TTSProcessor._model_cache = tts
                print("✅ TTS Model loaded successfully.")
            except Exception as e:
                print(f"❌ Error loading TTS Model: {e}")
                if sys.platform == 'win32': 
                    print("Hint: On Windows, enable 'Long Paths'.")
                raise
        else:
            print("✅ Using cached TTS Model.")
            
        self.tts = TTSProcessor._model_cache

    def force_split(self, text, max_chars):
        """
        Aggressively splits text if standard sentence splitting fails.
        Order of operations: Commas -> Spaces -> Hard Cut.
        """
        # 1. Try splitting by commas/semicolons
        parts = re.split(r'(?<=[,;:])\s+', text)
        
        # 2. If that didn't help (still one huge chunk), split by spaces
        if len(parts) == 1 and len(text) > max_chars:
            parts = text.split(' ')
            
        final_chunks = []
        current_chunk = ""
        
        for part in parts:
            if len(current_chunk) + len(part) < max_chars:
                current_chunk += part + " "
            else:
                if current_chunk:
                    final_chunks.append(current_chunk.strip())
                current_chunk = part + " "
                
                # 3. Nuclear Option: If a single part is STILL massive (garbage text), hard slice it
                while len(current_chunk) > max_chars:
                    final_chunks.append(current_chunk[:max_chars])
                    current_chunk = current_chunk[max_chars:]

        if current_chunk:
            final_chunks.append(current_chunk.strip())
            
        return final_chunks

    def split_text_into_chunks(self, text, max_chars=250):
        """
        Smart splitter that handles run-on sentences.
        """
        if len(text) <= max_chars:
            return [text]
            
        # 1. Try standard sentence delimiters
        raw_sentences = re.split(r'(?<=[.?!])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sent in raw_sentences:
            if not sent.strip(): continue
            
            # If a single "sentence" is huge, force split it
            if len(sent) > max_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                # Call the aggressive splitter
                sub_chunks = self.force_split(sent, max_chars)
                chunks.extend(sub_chunks)
                
            elif len(current_chunk) + len(sent) < max_chars:
                current_chunk += sent + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sent + " "
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    def generate_audio(self, text, speaker_wav_path, output_path, language="en"):
        if not os.path.exists(speaker_wav_path):
            raise FileNotFoundError(f"Speaker ref not found: {speaker_wav_path}")
            
        # Chunking Strategy
        chunks = self.split_text_into_chunks(text)
        
        if len(chunks) > 1:
            print(f"   ⚠️ Long text ({len(text)} chars). Split into {len(chunks)} safe chunks.")

        if len(chunks) == 1:
            # FAST PATH
            try:
                self.tts.tts_to_file(
                    text=chunks[0],
                    speaker_wav=speaker_wav_path,
                    language=language,
                    file_path=output_path
                )
                return output_path
            except Exception as e:
                print(f"⚠️ Standard TTS failed: {e}. Trying fallback...")

        # MERGE PATH (Safe)
        combined_audio = AudioSegment.empty()
        temp_files = []
        
        try:
            for i, chunk in enumerate(chunks):
                if not chunk.strip(): continue
                temp_file = f"{output_path}_part{i}.wav"
                temp_files.append(temp_file)
                
                self.tts.tts_to_file(
                    text=chunk,
                    speaker_wav=speaker_wav_path,
                    language=language,
                    file_path=temp_file
                )
                
                part_audio = AudioSegment.from_file(temp_file)
                combined_audio += part_audio
                
            combined_audio.export(output_path, format="wav")
            print(f"💾 Merged Audio saved to: {output_path}")
            
        except Exception as e:
            print(f"❌ Error during Chunk/Merge: {e}")
            raise
        finally:
            for f in temp_files:
                if os.path.exists(f):
                    os.remove(f)

        return output_path

if __name__ == "__main__":
    print("Robust TTS Module Ready.")