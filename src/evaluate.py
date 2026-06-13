# src/evaluate.py
"""Test the model on unseen machine types (zero-shot anomaly detection)"""

import os
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import config
from data_loader import AudioDataset
from features import batch_audio_to_spectrograms
from model import SimpleAutoencoder

def evaluate_on_unseen_domain(model, device, domain_name):
    """
    Evaluate model on a domain it hasn't seen during training
    """
    print(f"\n{'='*50}")
    print(f"Evaluating on unseen domain: {domain_name}")
    print(f"{'='*50}")
    
    # Use MIMII path directly
    data_path = os.path.join(config.DATA_ROOT, "MIMII")
    
    # Check if data exists
    if not os.path.exists(data_path):
        print(f"Error: Data path not found: {data_path}")
        return None
    
    # Load datasets
    normal_dataset = AudioDataset(data_path, domain_name, is_normal=True)
    anomaly_dataset = AudioDataset(data_path, domain_name, is_normal=False)
    
    if len(normal_dataset) == 0:
        print(f"Error: No normal files found for {domain_name}")
        return None
    
    if len(anomaly_dataset) == 0:
        print(f"Error: No anomaly files found for {domain_name}")
        return None
    
    print(f"Found {len(normal_dataset)} normal files")
    print(f"Found {len(anomaly_dataset)} anomaly files")
    
    # Compute anomaly scores
    normal_scores = []
    anomaly_scores = []
    
    model.eval()
    
    print("Processing normal samples...")
    num_normal = min(100, len(normal_dataset))
    for i in tqdm(range(num_normal)):
        audio = normal_dataset[i]
        spectrogram = batch_audio_to_spectrograms(
            torch.FloatTensor([audio]), 
            config.SAMPLE_RATE, 
            config.N_MELS
        )
        spectrogram = spectrogram.to(device)
        score = model.compute_anomaly_score(spectrogram)
        normal_scores.append(score)
    
    print("Processing anomaly samples...")
    num_anomaly = min(100, len(anomaly_dataset))
    for i in tqdm(range(num_anomaly)):
        audio = anomaly_dataset[i]
        spectrogram = batch_audio_to_spectrograms(
            torch.FloatTensor([audio]), 
            config.SAMPLE_RATE, 
            config.N_MELS
        )
        spectrogram = spectrogram.to(device)
        score = model.compute_anomaly_score(spectrogram)
        anomaly_scores.append(score)
    
    # Calculate AUC (Area Under Curve)
    y_true = [0] * len(normal_scores) + [1] * len(anomaly_scores)
    y_scores = normal_scores + anomaly_scores
    
    auc = roc_auc_score(y_true, y_scores)
    print(f"\nAUC for {domain_name}: {auc:.4f}")
    
    # Create results directory
    os.makedirs('../results/plots', exist_ok=True)
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {domain_name}', fontsize=14)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(f'../results/plots/roc_{domain_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to ../results/plots/roc_{domain_name}.png")
    
    # Plot score distributions
    plt.figure(figsize=(10, 5))
    plt.hist(normal_scores, bins=20, alpha=0.5, label='Normal', density=True, color='blue')
    plt.hist(anomaly_scores, bins=20, alpha=0.5, label='Anomaly', density=True, color='red')
    plt.xlabel('Anomaly Score (Reconstruction Error)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(f'Score Distribution - {domain_name}', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(f'../results/plots/scores_{domain_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Score distribution saved to ../results/plots/scores_{domain_name}.png")
    
    return {
        'domain': domain_name,
        'auc': auc,
        'normal_scores': normal_scores,
        'anomaly_scores': anomaly_scores
    }

def run_full_evaluation():
    """Run evaluation on all unseen domains"""
    
    # Set device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load trained model
    model = SimpleAutoencoder()
    model_path = "../results/models/final_model.pth"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please run train.py first.")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    
    # Evaluate on unseen domains
    results = []
    for domain in config.TARGET_DOMAINS:
        result = evaluate_on_unseen_domain(model, device, domain)
        if result:
            results.append(result)
    
    if len(results) == 0:
        print("\nNo results generated. Check if data exists.")
        return
    
    # Summary
    print("\n" + "="*50)
    print("FINAL RESULTS - Zero-Shot Anomaly Detection")
    print("="*50)
    for result in results:
        print(f"{result['domain']:20} AUC: {result['auc']:.4f}")
    
    # Calculate average AUC
    avg_auc = np.mean([r['auc'] for r in results])
    print(f"\n{'Average':20} AUC: {avg_auc:.4f}")
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    domains = [r['domain'] for r in results]
    aucs = [r['auc'] for r in results]
    
    bars = plt.bar(domains, aucs, color='steelblue', edgecolor='black')
    plt.xlabel('Unseen Domain', fontsize=12)
    plt.ylabel('AUC Score', fontsize=12)
    plt.title('Zero-Shot Anomaly Detection Performance', fontsize=14)
    plt.ylim(0, 1)
    plt.axhline(y=0.5, color='red', linestyle='--', label='Random Chance (0.5)')
    
    for bar, auc in zip(bars, aucs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{auc:.3f}', ha='center', fontsize=11)
    
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig('../results/plots/summary_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSummary plot saved to ../results/plots/summary_results.png")
    
    print("\n" + "="*50)
    print("Evaluation complete! Check ../results/plots/ for visualizations.")
    print("="*50)

if __name__ == "__main__":
    run_full_evaluation()