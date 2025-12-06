# training/hparams.py

hparams = {
    # --- Audio Parameters ---
    # The paper uses 80 filter banks for the Mel-spectrogram [cite: 255]
    "num_mels": 80,
    "rescale": True,
    "rescaling_max": 0.9,
    "use_lws": False,
    "n_fft": 800,
    "hop_size": 200,
    "win_size": 800,
    # The paper explicitly states 16kHz audio sampling [cite: 281]
    "sample_rate": 16000,
    "signal_normalization": True,
    "allow_clipping_in_normalization": True,
    "symmetric_mels": True,
    "max_abs_value": 4.0,
    "preemphasize": True,
    "preemphasis": 0.97,

    # --- Training Hyperparameters ---
    # The paper uses Adam optimizer with LR 5e-4 for Generator and 1e-4 for Discriminator 
    "initial_learning_rate": 5e-4, 
    "nepochs": 200000000000000000, # Run until manually stopped
    
    # Checkpointing
    "checkpoint_interval": 1000, # Save more often since epochs are slower at 256p
    "eval_interval": 1000,
    
    # CRITICAL: Batch Size
    # The paper uses 64 (distributed). For a single Kaggle GPU at 256x256, 
    # we must drop to 4 or 8 to avoid OOM.
    "batch_size": 8,  
    
    # --- Loss Weights ---
    # Standard Wav2Lip weights. 
    # (Note: The paper uses different weights: Rec=1, Land=100, GAN=1e-4[cite: 360],
    # but that requires changing the loss calculation code in train.py first.)
    "syncnet_wt": 0.03, # Lip Sync Weight
    "disc_wt": 0.07,    # GAN/Realism Weight
    
    # --- Visual Parameters ---
    # UPDATED: Changed from 96 to 256 to match the paper's resolution 
    "img_size": 256,
    
    # Frame shift usually 10ms-40ms depending on video FPS
    "frame_shift": 10 
}