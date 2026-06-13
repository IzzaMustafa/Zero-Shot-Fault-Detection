# src/features.py
import numpy as np
import librosa
import torch

def audio_to_spectrogram(audio, sample_rate=16000, n_mels=128, hop_length=512, fixed_length=128):
    """
    Convert audio to fixed-size 128x128 log-mel spectrogram
    """
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=n_mels,
        hop_length=hop_length,
        n_fft=hop_length * 2
    )
    
    # Convert to log scale
    log_mel = librosa.power_to_db(mel_spec + 1e-10)
    
    # Resize to exactly 128 time frames
    if log_mel.shape[1] > fixed_length:
        log_mel = log_mel[:, :fixed_length]
    elif log_mel.shape[1] < fixed_length:
        # Pad with zeros
        pad_width = fixed_length - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='constant')
    
    # Normalize to [0, 1]
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)
    
    # Add channel dimension: (1, 128, 128)
    spectrogram = torch.FloatTensor(log_mel).unsqueeze(0)
    
    return spectrogram

def batch_audio_to_spectrograms(audio_batch, sample_rate=16000, n_mels=128):
    """Convert batch of audio to spectrograms"""
    spectrograms = []
    for audio in audio_batch:
        if torch.is_tensor(audio):
            audio = audio.numpy()
        spec = audio_to_spectrogram(audio, sample_rate, n_mels)
        spectrograms.append(spec)
    
    return torch.stack(spectrograms)