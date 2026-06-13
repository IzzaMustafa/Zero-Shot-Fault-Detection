# Zero-Shot Fault Detection Across Unseen Industrial Domains

## 🎯 Overview

Deep learning model for zero-shot anomaly detection on industrial machine sounds. The model is trained on normal operating sounds from valve, pump, and fan machines and evaluated on an unseen machine domain (slider) without any target-domain training data.

## 🌍 Zero-Shot Generalization

The model is trained exclusively on source domains (Valve, Pump, Fan) and tested on an unseen target domain (Slider). No labeled or unlabeled slider samples are used during training, ensuring a true zero-shot evaluation setting.

## 📊 Key Results

* **Zero-Shot AUC Score: 0.8099 (81.0%)**
* Strong anomaly detection performance on an unseen domain
* Successful domain generalization without target-domain training data

## 🏗️ Architecture

* **Feature Extraction**: Log-mel spectrograms (128×128)
* **Model**: Convolutional Autoencoder
* **Training**: Only normal sounds from 3 domains
* **Anomaly Scoring**: Reconstruction error (MSE)

## 📈 Results Visualization

| Domain          | AUC Score  |
| --------------- | ---------- |
| Slider (Unseen) | **0.8099** |

### ROC Curve

### Score Distribution

### Training Loss

## 🔬 Ablation Study

| Model Variant          | AUC    |
| ---------------------- | ------ |
| Full Model             | 0.4136 |
| No BatchNorm           | 0.3964 |
| Single Domain Training | 0.3504 |

## 🚀 Quick Start

## Train Model

```bash
cd src
python train.py
```

## Evaluate on Unseen Domain

```bash
python evaluate.py
```

## Run Ablation Study

```bash
python ablation.py
```

## 📂 Dataset

* **MIMII Dataset**: Valve, Pump, Fan (training) + Slider (unseen test)
* **Sample Rate**: 16 kHz
* **Audio Length**: 10 seconds per sample

## 🔧 Requirements

* Python 3.8+
* PyTorch
* Librosa
* NumPy
* Scikit-learn
* Matplotlib
* SoundFile
* Tqdm

Install dependencies:

```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```text
Zero-Shot-Fault-Detection/
├── src/
│   ├── config.py          # Configuration settings
│   ├── data_loader.py     # Audio loading utilities
│   ├── features.py        # Spectrogram extraction
│   ├── model.py           # Autoencoder architecture
│   ├── train.py           # Training script
│   ├── evaluate.py        # Zero-shot evaluation
│   └── ablation.py        # Ablation study
├── results/
│   └── plots/             # Generated visualizations
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

## 📝 Results Interpretation

* **AUC = 0.8099**: The model distinguishes between normal and anomalous machine sounds with good accuracy on an unseen machine domain.
* **ROC Curve**: A curve closer to the top-left corner indicates stronger anomaly detection performance.
* **Score Distribution**: Separation between normal and anomaly reconstruction errors demonstrates effective anomaly detection capability.

## 🎯 Methodology

### Training Phase

The autoencoder is trained on normal machine sounds from:

* Valve
* Pump
* Fan

The model learns a compressed representation of normal operating behavior.

### Zero-Shot Testing

The trained model is evaluated on:

* Slider (unseen machine domain)

No slider samples are used during training.

### Anomaly Detection

Anomaly scores are computed using reconstruction error:

* Low reconstruction error → Normal sample
* High reconstruction error → Anomalous sample

This enables zero-shot fault detection on previously unseen machine types.

## 📊 Experimental Setup

* Feature Extraction: Log-Mel Spectrograms
* Model: Convolutional Autoencoder
* Loss Function: Mean Squared Error (MSE)
* Evaluation Metric: ROC-AUC
* Zero-Shot Protocol: Leave-One-Domain-Out

## 👩‍💻 Author

**Izza Mustafa Jadoon**