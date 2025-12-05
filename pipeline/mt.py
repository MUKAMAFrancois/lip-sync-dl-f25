# pip install transformers -U
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

class MTProcessor:
    # Singleton Cache
    _model_cache = None
    _tokenizer_cache = None
    
    # NLLB Language Codes Mapping
    # We map "simple" names to NLLB's specific codes
    LANG_CODES = {
        "english": "eng_Latn",
        "german": "deu_Latn",
        "french": "fra_Latn",
        "spanish": "spa_Latn",
        "russian": "rus_Cyrl",
        "italian": "ita_Latn"
    }

    def __init__(self, model_id="facebook/nllb-200-distilled-600M"):
        """
        Initializes the NLLB Translation Model.
        """
        self.device = 0 if torch.cuda.is_available() else -1
        self.device_name = "cuda" if torch.cuda.is_available() else "cpu"

        if MTProcessor._model_cache is None:
            print(f"...Loading MT Model ({model_id}) on {self.device_name}...")
            try:
                # Load Tokenizer & Model
                MTProcessor._tokenizer_cache = AutoTokenizer.from_pretrained(model_id)
                MTProcessor._model_cache = AutoModelForSeq2SeqLM.from_pretrained(model_id)
                print("Good: MT Model loaded successfully.")
            except Exception as e:
                print(f"!!! Error loading MT Model: {e}")
                raise
        else:
            print(f"Good: Using cached MT Model ({model_id}).")

        # Create the pipeline using the cached model
        # Note: We don't specify task="translation_xx_to_yy" because NLLB is many-to-many
        self.pipe = pipeline(
            "translation",
            model=MTProcessor._model_cache,
            tokenizer=MTProcessor._tokenizer_cache,
            device=self.device
        )

    def translate(self, text, source_lang="german", target_lang="english"):
        """
        Translates text using NLLB.
        Args:
            text (str): Input text.
            source_lang (str): Source language name (e.g., "german").
            target_lang (str): Target language name (e.g., "english").
        """
        # Convert simple names to NLLB codes (e.g., "german" -> "deu_Latn")
        src_code = self.LANG_CODES.get(source_lang.lower())
        tgt_code = self.LANG_CODES.get(target_lang.lower())

        if not src_code or not tgt_code:
            raise ValueError(f"Unsupported language pair: {source_lang} -> {target_lang}")

        # NLLB requires explicit source language definition
        # We pass this via generate_kwargs
        output = self.pipe(
            text, 
            src_lang=src_code, 
            tgt_lang=tgt_code,
            max_length=512
        )
        
        translated_text = output[0]['translation_text']
        print(f"Translated ({source_lang}->{target_lang}): '{text[:30]}...' -> '{translated_text[:30]}...'")
        return translated_text

# --- Testing Block ---
if __name__ == "__main__":
    # Test on your sample German text
    sample_text = "So schön mal wieder hier in Tübingen zu sein."
    
    mt = MTProcessor()
    
    # 1st Call (Should Load)
    translation = mt.translate(sample_text, source_lang="german", target_lang="english")
    print(f"\nResult: {translation}")

    # 2nd Call (Should Cache) - Try French just to prove it handles multiple langs
    sample_french = "Bonjour tout le monde."
    translation_fr = mt.translate(sample_french, source_lang="french", target_lang="english")
    print(f"Result: {translation_fr}")