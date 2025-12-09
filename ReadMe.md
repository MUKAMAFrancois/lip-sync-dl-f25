# Project Summary

**Project Title:** Voice-Preserving Multilingual Video Translator Using NLP and AI Voice
Cloning

**Overview:**
This project implements an end-to-end automated dubbing pipeline capable of translating German video content into English while maintaining high-fidelity audio-visual synchronization. Unlike traditional dubbing, which suffers from the "uncanny valley" effect due to mismatched lip movements, this system employs a generative approach to visual synthesis.

**Methodology:**
The system operates on a "Dub and Sync" architecture. The **Audio Pipeline** leverages SOTA models (Whisper, NLLB, Coqui XTTS) to transcribe, translate, and clone the original speaker's voice. A novel **Elastic Duration Aligner** dynamically compresses or positions the translated audio to strictly adhere to the temporal constraints of the source video. The **Visual Pipeline** utilizes a modified Wav2Lip GAN, fine-tuned on high-resolution (256x256) facial crops, to generate lip movements that are synchronized with the new English phonemes, effectively "re-animating" the speaker's face.

-----


# System Architecture

The pipeline consists of two main stages:

### 1. Audio Pipeline (The "Dub")
* **Source Separation:** Separates vocal tracks from background music using **Demucs**.
* **ASR:** Transcribes German speech with timestamp precision using **OpenAI Whisper (Medium)**.
* **Translation:** Translates text to English using **NLLB-200**.
* **Voice Cloning:** Synthesizes English speech with the original speaker's timbre using **Coqui XTTS v2**.
* **Elastic Alignment:** A custom `DurationAligner` dynamically time-stretches and positions audio to fit original video segments.

### 2. Visual Pipeline (The "Sync")
* **Face Alignment:** Detects and aligns faces to 256x256 resolution using **S3FD**.
* **Lip Synthesis:** Generates new mouth movements using a **Wav2Lip GAN** fine-tuned on the target speaker.
* **Rendering:** Blends the generated lips back into the original video frames.

##  Installation

This project is designed to run on Linux environments (Colab, Kaggle, PSC Bridges-2) with GPU support.

### Prerequisites
* Python 3.10+
* FFmpeg
* NVIDIA GPU (L4/V100 recommended)

### Setup
```bash
# 1. Clone the repository
git clone https://github.com/MUKAMAFrancois/lip-sync-dl-f25.git
cd lip-sync-dl-f25

# 2. Install Dependencies (Order matters to avoid version conflicts)
pip uninstall -y numpy
pip install "numpy<2.0"
pip install -r requirements.txt
# Install specific versions for stability
pip install face-alignment pytorch-fid librosa==0.10.1 --no-deps

# 3. Setup Wav2Lip & Download Weights
python setup_wav2lip.py
````

## Usage Guide

### Phase 1: Preprocessing

Extracts audio and prepares aligned face crops for training.

```bash
python preprocess_training_data.py \
  --data_root data/german \
  --output_root data/german/preprocessed \
  --filelist_root data/german/filelists \
  --no-zip
```

### Phase 2: Training (Fine-Tuning)

Fine-tunes the Lip-Sync model on your dataset.

  * **Note:** Ensure you check `training/hparams.py` to set `batch_size` according to your GPU memory (Default: 8).

<!-- end list -->

```bash
python training/train.py
```

  * *Output:* Checkpoints will be saved in `checkpoints/`.

### Phase 3: Inference (Full Pipeline)

Runs the complete Dubbing Pipeline (ASR -\> MT -\> TTS -\> Align -\> Wav2Lip) on new videos.

```bash
python pipeline/main.py \
    --input_path "data/german/test" \
    --output_root "results/dubbed_test" \
    --source_lang "german"
```

  * *Input:* A folder of `.mp4` videos.
  * *Output:* Dubbed videos saved in `results/dubbed_test`.

### Phase 4: Evaluation

Calculates quantitative metrics (SSIM, PSNR, LMD, FID) to assess quality.

```bash
python evaluate.py \
    --gt_path "data/german/preprocessed/val" \
    --gen_path "results/dubbed_test"
```

##  Metrics Explained

| Metric | Description | Target |
| :--- | :--- | :--- |
| **SSIM** | Structural Similarity (Visual Quality) | \> 0.8 |
| **PSNR** | Peak Signal-to-Noise Ratio (Reconstruction) | \> 30.0 dB |
| **LMD** | Landmark Distance (Lip Sync Accuracy) | \< 3.5 |
| **FID** | Fréchet Inception Distance (Realism) | Lower is better |

##  File Structure

```
lip-sync-dl-f25/
├── pipeline/               # Core Logic
│   ├── main.py             # Pipeline Orchestrator
│   ├── inference.py        # Visual Dubbing (Wav2Lip)
│   ├── duration_aligner.py # Elastic Audio Sync
│   ├── tts.py              # Voice Cloning
│   └── ...
├── training/               # Training Logic
│   ├── train.py            # Training Loop
│   ├── data_loader.py      # Dataset & Masking
│   └── hparams.py          # Config
├── preprocess_training_data.py
├── evaluate.py
└── requirements.txt
```

##  Acknowledgements

  * https://arxiv.org/abs/2011.03530

<!-- end list -->
