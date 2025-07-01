#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2025 Agora
# This file is part of TEN Framework, an open source project.
# Licensed under the Apache License, Version 2.0, with certain conditions.
# Refer to the "LICENSE" file in the root directory for more information.
#

import numpy as np
import torch
import os
import sys
from pathlib import Path
import tempfile
import shutil

# Add the current directory to Python path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vad_trainer import VADTrainer, VADDataset, VADFeatureExtractor
from ten_vad_reconstructed import TenVAD
from prepare_vad_data import prepare_dataset


def create_synthetic_audio_files(num_files=10, duration=10.0, output_dir="synthetic_data"):
    """
    Create synthetic audio files for testing
    
    Args:
        num_files: Number of audio files to create
        duration: Duration of each file in seconds
        output_dir: Output directory
    """
    import scipy.io.wavfile as wavfile
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    sample_rate = 16000
    samples_per_file = int(duration * sample_rate)
    
    print(f"Creating {num_files} synthetic audio files...")
    
    for i in range(num_files):
        # Generate random audio (mix of speech-like and noise)
        t = np.linspace(0, duration, samples_per_file)
        
        # Speech-like signal (sine wave with varying frequency)
        speech_freq = 200 + 100 * np.sin(2 * np.pi * 0.5 * t)
        speech_signal = 0.3 * np.sin(2 * np.pi * speech_freq * t)
        
        # Add some noise
        noise = 0.1 * np.random.randn(samples_per_file)
        
        # Combine and normalize
        audio = speech_signal + noise
        audio = np.clip(audio, -1.0, 1.0)
        
        # Convert to int16
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Save file
        filename = output_path / f"synthetic_audio_{i:03d}.wav"
        wavfile.write(filename, sample_rate, audio_int16)
    
    print(f"Synthetic audio files created in {output_dir}")


def run_training_example():
    """Run a complete training example with synthetic data"""
    
    # Create synthetic data
    print("=== Creating Synthetic Data ===")
    create_synthetic_audio_files(num_files=20, duration=5.0, output_dir="synthetic_data")
    
    # Prepare dataset with labels
    print("\n=== Preparing Dataset ===")
    prepare_dataset("synthetic_data", "vad_dataset", method='energy', train_split=0.8)
    
    # Initialize components
    print("\n=== Initializing Training Components ===")
    feature_extractor = VADFeatureExtractor()
    
    # Create datasets
    train_dataset = VADDataset("vad_dataset/train", feature_extractor, max_sequence_length=500)
    val_dataset = VADDataset("vad_dataset/val", feature_extractor, max_sequence_length=500)
    
    print(f"Training dataset: {len(train_dataset)} samples")
    print(f"Validation dataset: {len(val_dataset)} samples")
    
    # Create data loaders
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    # Initialize model and trainer
    print("\n=== Initializing Model and Trainer ===")
    model = TenVAD()
    trainer = VADTrainer(model, device='cuda' if torch.cuda.is_available() else 'cpu')
    trainer.setup_optimizer(learning_rate=1e-3, weight_decay=1e-4)
    
    # Train for a few epochs
    print("\n=== Starting Training ===")
    trainer.train(train_loader, val_loader, num_epochs=5, save_dir='example_checkpoints')
    
    print("\n=== Training Complete ===")
    print("Checkpoints saved in 'example_checkpoints' directory")
    
    # Test inference on a sample
    print("\n=== Testing Inference ===")
    test_inference(trainer.model, feature_extractor)


def test_inference(model, feature_extractor):
    """Test inference on a sample audio file"""
    import scipy.io.wavfile as wavfile
    
    # Find a test file
    test_files = list(Path("synthetic_data").glob("*.wav"))
    if not test_files:
        print("No test files found")
        return
    
    test_file = test_files[0]
    print(f"Testing inference on {test_file}")
    
    # Load audio
    sample_rate, audio_data = wavfile.read(test_file)
    
    # Convert to float32
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32)
    
    # Extract features
    features = feature_extractor.extract_features_from_audio(audio_data)
    
    # Create context windows
    from vad_trainer import CONTEXT_WINDOW_LEN, FEATURE_LEN
    context_features = []
    
    for i in range(CONTEXT_WINDOW_LEN - 1, len(features)):
        context_window = features[i - CONTEXT_WINDOW_LEN + 1:i + 1]
        context_features.append(context_window)
    
    if not context_features:
        print("No valid context windows found")
        return
    
    # Convert to tensor
    input_tensor = torch.from_numpy(np.array(context_features[:10])).float()  # Test first 10 frames
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
    
    # Move to device
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # Initialize hidden states
    from vad_trainer import MODEL_HIDDEN_DIM, MODEL_IO_NUM
    hidden_states = [torch.zeros(1, 1, MODEL_HIDDEN_DIM, device=device) 
                     for _ in range(MODEL_IO_NUM - 1)]
    
    # Run inference
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor, *hidden_states)
        predictions = outputs[0].squeeze().cpu().numpy()
    
    print(f"VAD predictions (first 10 frames):")
    for i, pred in enumerate(predictions):
        print(f"  Frame {i}: {pred:.4f} ({'speech' if pred > 0.5 else 'silence'})")


def main():
    """Main function"""
    print("TEN VAD Training Example")
    print("=" * 50)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, using CPU")
    
    try:
        run_training_example()
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    print("\n=== Cleanup ===")
    cleanup_dirs = ["synthetic_data", "vad_dataset", "example_checkpoints"]
    for dir_name in cleanup_dirs:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"Removed {dir_name}")


if __name__ == "__main__":
    main() 