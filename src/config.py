# src/config.py
"""Configuration settings for the project"""

class Config:
    # Data settings
    DATA_ROOT = "../data/"
    SAMPLE_RATE = 16000  # Audio sample rate (Hz)
    DURATION = 10  # Length of each audio clip (seconds)
    
    # Feature extraction
    N_MELS = 128  # Number of mel bands (makes 128x1280 spectrograms)
    HOP_LENGTH = 512  # Hop length for STFT
    
    # Model settings
    LATENT_DIM = 64  # Size of the bottleneck
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    
    # Training
    DEVICE = "cuda"  # Use "cuda" if you have GPU, else "cpu"
    SEED = 42  # For reproducibility
    
    # Machine types
    SOURCE_DOMAINS = ["valve", "pump", "fan"]   # Train on 3 machines
    TARGET_DOMAINS = ["slider"] 
    
config = Config()