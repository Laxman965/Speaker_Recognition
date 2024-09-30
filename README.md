# Speaker Recognition System

## Overview

The Speaker Recognition System is a machine learning project designed to identify speakers from audio clips. This project leverages deep learning techniques, specifically Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks, to effectively recognize and classify different speakers based on their unique vocal characteristics.

## Features

- **Speaker Identification**: Recognizes and classifies speakers from audio data.
- **High Accuracy**: Utilizes state-of-the-art deep learning models for improved accuracy.
- **Custom Dataset**: Created a unique dataset with 900 training samples for robust model training.
- **User-friendly Interface**: Easy to integrate and use for further applications.

## Technologies Used

- Python
- TensorFlow/Keras
- Librosa (for audio processing)
- NumPy
- Matplotlib (for visualization)
- Jupyter Notebook (for development)

## Data Collection

The dataset for this project was created by embedding 2-second speaker clips into 10-second audio noise clips. This approach helps simulate real-world conditions where background noise may affect recognition performance.

- **Training Samples**: 900 samples
- **Sample Duration**: 10 seconds each (2 seconds of speech and 8 seconds of noise)

## Model Architecture

The model consists of the following layers:

1. **Convolutional Layers**: Used for feature extraction from the audio signals.
2. **MaxPooling Layers**: Used for downsampling the feature maps to reduce dimensionality.
3. **LSTM Layers**: Designed to learn temporal patterns in the audio sequences.
4. **Dense Layer**: Outputs the final predictions for speaker classification.

### Model Compilation

- **Loss Function**: `binary_crossentropy`
- **Optimizer**: `adam`
- **Metrics**: `accuracy`, `Precision`, `AUC`, `Recall`, `mse`

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Speaker-Recognition-System.git
   cd Speaker-Recognition-System
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
3. Prepare your dataset:  
    Place your audio dataset in the designated folder (update the paths in the scripts accordingly).
4. Run the training script:
   ```bash
   python train.py
5. Evaluate the model:
   ```bash
   python evaluate.py


## Results

The model achieved significant performance with the following metrics:

- **Accuracy**: 86.96%  
- **Precision**: 87.7%  
- **Recall**: 71.29%  
- **AUC**: 89.91%

