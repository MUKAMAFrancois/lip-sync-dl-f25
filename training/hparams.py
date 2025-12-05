hparams = {
    # Audio Params
    "num_mels": 80,
    "rescale": True,
    "rescaling_max": 0.9,
    "use_lws": False,
    "n_fft": 800,
    "hop_size": 200,
    "win_size": 800,
    "sample_rate": 16000,
    "signal_normalization": True,
    "allow_clipping_in_normalization": True,
    "symmetric_mels": True,
    "max_abs_value": 4.0,
    "preemphasize": True,
    "preemphasis": 0.97,

    # Training Params
    "initial_learning_rate": 1e-4,
    "nepochs": 200000000000000000, # Infinite, we stop manually
    "checkpoint_interval": 3000,
    "eval_interval": 3000,
    "batch_size": 16,  # 16 is safe for Colab T4 GPU
    "syncnet_wt": 0.03, # Importance of Lip Sync
    "disc_wt": 0.07,    # Importance of Realism (GAN)
    
    # Model
    "img_size": 96,
    "frame_shift": 10 # For temporal continuity
}