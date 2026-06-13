# src/data_loader.py
"""Load audio files from MIMII and ToyADMOS datasets"""

import os
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from config import config

class AudioDataset(Dataset):
    """
    Loads audio files from MIMII dataset with id_XX subfolders
    """
    def __init__(self, data_dir, machine_type, is_normal=True):
        """
        Args:
            data_dir: Path to dataset folder (e.g., "../data/MIMII/")
            machine_type: e.g., "valve", "pump", "fan", "slider"
            is_normal: True for normal sounds, False for anomaly sounds
        """
        self.sample_rate = config.SAMPLE_RATE
        self.duration = config.DURATION
        self.samples_per_file = config.SAMPLE_RATE * config.DURATION
        
        # Find all audio files
        self.file_paths = []
        
        # MIMII structure: data/MIMII/slider/id_00/normal/*.wav
        machine_path = os.path.join(data_dir, machine_type)
        
        if os.path.exists(machine_path):
            # Look for id_XX folders
            for item in os.listdir(machine_path):
                item_path = os.path.join(machine_path, item)
                if os.path.isdir(item_path) and item.startswith('id_'):
                    # Found an id folder
                    if is_normal:
                        audio_path = os.path.join(item_path, "normal")
                    else:
                        # Check both possible folder names
                        anomaly_path = os.path.join(item_path, "anomaly")
                        abnormal_path = os.path.join(item_path, "abnormal")
                        
                        if os.path.exists(anomaly_path):
                            audio_path = anomaly_path
                        elif os.path.exists(abnormal_path):
                            audio_path = abnormal_path
                        else:
                            audio_path = None
                    
                    if os.path.exists(audio_path):
                        for f in os.listdir(audio_path):
                            if f.endswith('.wav'):
                                self.file_paths.append(os.path.join(audio_path, f))
        
        print(f"Found {len(self.file_paths)} {'normal' if is_normal else 'anomaly'} files in {machine_path}")
        
        if len(self.file_paths) == 0:
            print(f"WARNING: No files found! Checked path: {machine_path}")
            if os.path.exists(machine_path):
                print(f"Contents of {machine_path}: {os.listdir(machine_path)}")
                # Check inside first id folder
                for item in os.listdir(machine_path):
                    item_path = os.path.join(machine_path, item)
                    if os.path.isdir(item_path):
                        print(f"  {item} contains: {os.listdir(item_path)}")
                        break
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Load audio file
        audio, sr = sf.read(self.file_paths[idx])
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Pad or truncate to fixed length
        if len(audio) < self.samples_per_file:
            # Pad with zeros
            audio = np.pad(audio, (0, self.samples_per_file - len(audio)))
        else:
            # Take first N samples
            audio = audio[:self.samples_per_file]
        
        return audio.astype(np.float32)

def create_dataloaders(machine_types, batch_size=32, is_normal=True):
    """
    Create dataloaders for multiple machine types
    Returns: dict of dataloaders
    """
    dataloaders = {}
    
    for machine_type in machine_types:
        # MIMII path
        mimii_path = os.path.join(config.DATA_ROOT, "MIMII")
        
        dataset = AudioDataset(mimii_path, machine_type, is_normal)
        
        if len(dataset) > 0:
            dataloaders[machine_type] = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=True,
                num_workers=0
            )
            print(f"Created dataloader for {machine_type} with {len(dataset)} files")
        else:
            print(f"WARNING: No files found for {machine_type}")
    
    return dataloaders