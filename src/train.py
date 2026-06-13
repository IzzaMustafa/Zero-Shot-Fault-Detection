# src/train.py
"""Train the autoencoder on normal sounds from multiple domains"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from config import config
from data_loader import create_dataloaders
from features import batch_audio_to_spectrograms
from model import SimpleAutoencoder

def train():
    """Main training function"""
    
    # Set random seed for reproducibility
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    
    # Set device (GPU if available)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataloaders for all source domains
    print("Loading data...")
    dataloaders = create_dataloaders(
        config.SOURCE_DOMAINS,
        batch_size=config.BATCH_SIZE,
        is_normal=True  # Only normal sounds for training
    )
    
    # Create model
    model = SimpleAutoencoder()
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error loss
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Training loop
    print("Starting training...")
    train_losses = []
    
    for epoch in range(config.NUM_EPOCHS):
        epoch_loss = 0
        num_batches = 0
        
        # Train on each domain separately (but same model)
        for domain_name, dataloader in dataloaders.items():
            model.train()
            
            for audio_batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} - {domain_name}"):
                # Convert audio to spectrograms
                spectrograms = batch_audio_to_spectrograms(
                    audio_batch, 
                    config.SAMPLE_RATE, 
                    config.N_MELS
                )
                spectrograms = spectrograms.to(device)
                
                # Forward pass
                reconstructed, latent = model(spectrograms)
                
                # Compute loss
                loss = criterion(reconstructed, spectrograms)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.6f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"../results/models/model_epoch_{epoch+1}.pth"
            os.makedirs("../results/models", exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_path = "../results/models/final_model.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Model saved to {final_path}")
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig('../results/plots/training_loss.png')
    plt.show()
    
    return model

if __name__ == "__main__":
    # Create results directories
    os.makedirs("../results/models", exist_ok=True)
    os.makedirs("../results/plots", exist_ok=True)
    
    # Train the model
    model = train()