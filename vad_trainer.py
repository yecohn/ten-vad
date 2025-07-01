#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2025 Agora
# This file is part of TEN Framework, an open source project.
# Licensed under the Apache License, Version 2.0, with certain conditions.
# Refer to the "LICENSE" file in the root directory for more information.
#

import numpy as np
import scipy.io.wavfile as wavfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import sys
import json
import logging
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from ten_vad_reconstructed import TenVAD

# Constants definition (corresponding to constants in C++ code)
SAMPLE_RATE = 16000
HOP_SIZE = 256  # 16ms per frame
FFT_SIZE = 1024
WINDOW_SIZE = 768
MEL_FILTER_BANK_NUM = 40
FEATURE_LEN = MEL_FILTER_BANK_NUM + 1  # 40 mel features + 1 pitch feature
CONTEXT_WINDOW_LEN = 3
MODEL_HIDDEN_DIM = 64
MODEL_IO_NUM = 5
EPS = 1e-20
PRE_EMPHASIS_COEFF = 0.97

# Feature normalization parameters (from coeff.h)
FEATURE_MEANS = np.array([
    -8.198236465454e+00, -6.265716552734e+00, -5.483818531036e+00,
    -4.758691310883e+00, -4.417088985443e+00, -4.142892837524e+00,
    -3.912850379944e+00, -3.845927953720e+00, -3.657090425491e+00,
    -3.723418712616e+00, -3.876134157181e+00, -3.843890905380e+00,
    -3.690405130386e+00, -3.756065845490e+00, -3.698696136475e+00,
    -3.650463104248e+00, -3.700468778610e+00, -3.567321300507e+00,
    -3.498900175095e+00, -3.477807044983e+00, -3.458816051483e+00,
    -3.444923877716e+00, -3.401328563690e+00, -3.306261301041e+00,
    -3.278556823730e+00, -3.233250856400e+00, -3.198616027832e+00,
    -3.204526424408e+00, -3.208798646927e+00, -3.257838010788e+00,
    -3.381376743317e+00, -3.534021377563e+00, -3.640867948532e+00,
    -3.726858854294e+00, -3.773730993271e+00, -3.804667234421e+00,
    -3.832901000977e+00, -3.871120452881e+00, -3.990592956543e+00,
    -4.480289459229e+00, 9.235690307617e+01
], dtype=np.float32)

FEATURE_STDS = np.array([
    5.166063785553e+00, 4.977209568024e+00, 4.698895931244e+00,
    4.630621433258e+00, 4.634347915649e+00, 4.641156196594e+00,
    4.640676498413e+00, 4.666367053986e+00, 4.650534629822e+00,
    4.640020847321e+00, 4.637400150299e+00, 4.620099067688e+00,
    4.596316337585e+00, 4.562654972076e+00, 4.554360389709e+00,
    4.566910743713e+00, 4.562489986420e+00, 4.562412738800e+00,
    4.585299491882e+00, 4.600179672241e+00, 4.592845916748e+00,
    4.585922718048e+00, 4.583496570587e+00, 4.626092910767e+00,
    4.626957893372e+00, 4.626289367676e+00, 4.637005805969e+00,
    4.683015823364e+00, 4.726813793182e+00, 4.734289646149e+00,
    4.753227233887e+00, 4.849722862244e+00, 4.869434833527e+00,
    4.884482860565e+00, 4.921327114105e+00, 4.959212303162e+00,
    4.996619224548e+00, 5.044823646545e+00, 5.072216987610e+00,
    5.096439361572e+00, 1.152136917114e+02
], dtype=np.float32)


class VADFeatureExtractor:
    """Feature extraction class for VAD training"""
    
    def __init__(self):
        # Pre-emphasis related
        self.pre_emphasis_prev = 0.0
        
        # Generate mel filter bank
        self.mel_filters = self._generate_mel_filters()
        
        # Window function (Hann window)
        self.window = np.hanning(WINDOW_SIZE).astype(np.float32)
    
    def _generate_mel_filters(self):
        """Generate mel filter bank"""
        n_bins = FFT_SIZE // 2 + 1
        
        # Generate mel frequency points
        low_mel = 2595.0 * np.log10(1.0 + 0.0 / 700.0)
        high_mel = 2595.0 * np.log10(1.0 + 8000.0 / 700.0)
        
        mel_points = np.linspace(low_mel, high_mel, MEL_FILTER_BANK_NUM + 2)
        hz_points = 700.0 * (10**(mel_points / 2595.0) - 1.0)
        
        # Convert to FFT bin indices
        bin_points = np.floor((FFT_SIZE + 1) * hz_points / SAMPLE_RATE).astype(int)
        
        # Build mel filter bank
        mel_filters = np.zeros((MEL_FILTER_BANK_NUM, n_bins), dtype=np.float32)
        
        for i in range(MEL_FILTER_BANK_NUM):
            # Left slope
            for j in range(bin_points[i], bin_points[i+1]):
                if j < n_bins:
                    mel_filters[i, j] = (j - bin_points[i]) / (bin_points[i+1] - bin_points[i])
            
            # Right slope
            for j in range(bin_points[i+1], bin_points[i+2]):
                if j < n_bins:
                    mel_filters[i, j] = (bin_points[i+2] - j) / (bin_points[i+2] - bin_points[i+1])
        
        return mel_filters
    
    def _pre_emphasis(self, audio_frame):
        """Pre-emphasis filtering"""
        emphasized = np.zeros_like(audio_frame)
        emphasized[0] = audio_frame[0] - PRE_EMPHASIS_COEFF * self.pre_emphasis_prev
        
        for i in range(1, len(audio_frame)):
            emphasized[i] = audio_frame[i] - PRE_EMPHASIS_COEFF * audio_frame[i-1]
        
        self.pre_emphasis_prev = audio_frame[-1]
        return emphasized
    
    def _extract_features(self, audio_frame):
        """
        Extract features from audio frame
        
        Args:
            audio_frame: Audio frame (256 samples)
        
        Returns:
            features: Feature vector (41 dimensions)
        """
        # Pre-emphasis
        emphasized = self._pre_emphasis(audio_frame)
        
        # Zero-padding to window size
        if len(emphasized) < WINDOW_SIZE:
            padded = np.zeros(WINDOW_SIZE)
            padded[:len(emphasized)] = emphasized
        else:
            padded = emphasized[:WINDOW_SIZE]
        
        # Windowing
        windowed = padded * self.window
        
        # FFT
        fft_result = np.fft.rfft(windowed, n=FFT_SIZE)
        
        # Compute power spectrum
        power_spectrum = np.abs(fft_result) ** 2
        
        # Normalization (corresponding to powerNormal = 32768^2 in C++)
        power_normal = 32768.0 ** 2
        power_spectrum = power_spectrum / power_normal
        
        # Mel filter bank features
        mel_features = np.dot(self.mel_filters, power_spectrum)
        mel_features = np.log(mel_features + EPS)
        
        # Simple pitch estimation (using 0 here, actual C++ code has complex pitch estimation)
        pitch_freq = 0.0
        
        # Combine features
        features = np.concatenate([mel_features, [pitch_freq]])
        
        # Feature normalization
        features = (features - FEATURE_MEANS) / (FEATURE_STDS + EPS)
        
        return features.astype(np.float32)
    
    def extract_features_from_audio(self, audio_data):
        """
        Extract features from entire audio file
        
        Args:
            audio_data: Audio data array
        
        Returns:
            features: Feature array (num_frames, FEATURE_LEN)
        """
        num_frames = len(audio_data) // HOP_SIZE
        features = []
        
        for i in range(num_frames):
            start_idx = i * HOP_SIZE
            end_idx = start_idx + HOP_SIZE
            
            if end_idx > len(audio_data):
                break
                
            audio_frame = audio_data[start_idx:end_idx]
            frame_features = self._extract_features(audio_frame)
            features.append(frame_features)
        
        return np.array(features)
    
    def reset(self):
        """Reset feature extractor state"""
        self.pre_emphasis_prev = 0.0


class VADDataset(Dataset):
    """Dataset for VAD training"""
    
    def __init__(self, data_dir, feature_extractor, max_sequence_length=1000):
        """
        Initialize VAD dataset
        
        Args:
            data_dir: Directory containing audio files and labels
            feature_extractor: VADFeatureExtractor instance
            max_sequence_length: Maximum sequence length for training
        """
        self.data_dir = Path(data_dir)
        self.feature_extractor = feature_extractor
        self.max_sequence_length = max_sequence_length
        
        # Find all audio files
        self.audio_files = list(self.data_dir.glob("*.wav"))
        
        if not self.audio_files:
            raise ValueError(f"No audio files found in {data_dir}")
        
        print(f"Found {len(self.audio_files)} audio files")
    
    def __len__(self):
        return len(self.audio_files)
    
    def _load_labels(self, audio_file):
        """Load VAD labels for audio file"""
        # Look for corresponding label file
        label_file = audio_file.with_suffix('.txt')
        
        labels = []
        with open(label_file, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('Start'):
                    parts = line.strip().split('\t')
                    if len(parts) >= 4:
                        start_time = float(parts[0])
                        end_time = float(parts[1])
                        label = 1 if parts[3] == 'speech' else 0
                        labels.append((start_time, end_time, label))
        
        return labels
    
    
    def _create_frame_labels(self, labels, num_frames):
        """Convert time-based labels to frame-based labels"""
        frame_labels = np.zeros(num_frames, dtype=np.float32)
        frame_duration = HOP_SIZE / SAMPLE_RATE
        
        for start_time, end_time, label in labels:
            start_frame = int(start_time / frame_duration)
            end_frame = int(end_time / frame_duration)
            frame_labels[start_frame:end_frame] = label
        
        return frame_labels
    
    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        
        # Load audio
        sample_rate, audio_data = wavfile.read(str(audio_file))
        
        # Convert to float32
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32)
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 65536.0
        
        # Extract features
        features = self.feature_extractor.extract_features_from_audio(audio_data)
        
        # Load labels
        labels = self._load_labels(audio_file)
        frame_labels = self._create_frame_labels(labels, len(features))
        
        # Truncate if too long
        if len(features) > self.max_sequence_length:
            features = features[:self.max_sequence_length]
            frame_labels = frame_labels[:self.max_sequence_length]
        
        # Create context windows
        context_features = []
        context_labels = []
        
        for i in range(CONTEXT_WINDOW_LEN - 1, len(features)):
            # Get context window
            context_window = features[i - CONTEXT_WINDOW_LEN + 1:i + 1]
            context_features.append(context_window)
            
            # Use the label of the last frame in the context window
            context_labels.append(frame_labels[i])
        
        if not context_features:
            # Handle very short sequences
            context_features = [np.zeros((CONTEXT_WINDOW_LEN, FEATURE_LEN), dtype=np.float32)]
            context_labels = [0.0]
        
        return {
            'features': torch.from_numpy(np.array(context_features)).float(),
            'labels': torch.from_numpy(np.array(context_labels)).float(),
            'filename': audio_file.name
        }


class VADTrainer:
    """VAD model trainer"""
    
    def __init__(self, model, device='cuda'):
        """
        Initialize VAD trainer
        
        Args:
            model: TenVAD model instance
            device: Device to use for training
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCELoss()
        self.optimizer = None
        self.scheduler = None
        
        # Training state
        self.epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Initialize hidden states
        self.hidden_states = [torch.zeros(1, 1, MODEL_HIDDEN_DIM, device=device) 
                             for _ in range(MODEL_IO_NUM - 1)]
    
    def setup_optimizer(self, learning_rate=1e-3, weight_decay=1e-4):
        """Setup optimizer and scheduler"""
        self.optimizer = optim.Adam(self.model.parameters(), 
                                   lr=learning_rate, 
                                   weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
    
    def reset_hidden_states(self):
        """Reset hidden states"""
        self.hidden_states = [torch.zeros(1, 1, MODEL_HIDDEN_DIM, device=self.device) 
                             for _ in range(MODEL_IO_NUM - 1)]
    
    def forward_pass(self, features, labels):
        """
        Forward pass with loss computation
        
        Args:
            features: Input features (batch_size, seq_len, context_window, feature_dim)
            labels: Target labels (batch_size, seq_len)
        
        Returns:
            loss: Computed loss
            predictions: Model predictions
        """
        batch_size, seq_len = features.shape[:2]
        total_loss = 0
        all_predictions = []
        
        # Process each sequence in the batch
        for b in range(batch_size):
            sequence_loss = 0
            sequence_predictions = []
            
            # Reset hidden states for each sequence
            self.reset_hidden_states()
            
            # Process each frame in the sequence
            for t in range(seq_len):
                # Prepare input
                input_features = features[b, t].unsqueeze(0)  # (1, context_window, feature_dim)
                
                # Forward pass
                outputs = self.model(input_features, *self.hidden_states)
                prediction = outputs[0]  # (1, 1, 1)
                
                # Update hidden states
                self.hidden_states = list(outputs[1:])
                
                # Compute loss for this frame
                target = labels[b, t].unsqueeze(0).unsqueeze(0).unsqueeze(0)
                frame_loss = self.criterion(prediction, target)
                sequence_loss += frame_loss
                
                sequence_predictions.append(prediction.squeeze().item())
            
            total_loss += sequence_loss / seq_len
            all_predictions.append(sequence_predictions)
        
        avg_loss = total_loss / batch_size
        return avg_loss, all_predictions
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.epoch + 1}")
        
        for batch in progress_bar:
            features = batch['features'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            
            loss, predictions = self.forward_pass(features, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                loss, predictions = self.forward_pass(features, labels)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def train(self, train_loader, val_loader, num_epochs, save_dir='checkpoints'):
        """Train the model"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        print(f"Starting training for {num_epochs} epochs")
        print(f"Training on {len(train_loader.dataset)} samples")
        print(f"Validating on {len(val_loader.dataset)} samples")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss = self.validate(val_loader)
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(save_dir / 'best_model.pth')
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(save_dir / f'checkpoint_epoch_{epoch + 1}.pth')
            
            print(f"Epoch {epoch + 1}/{num_epochs}: "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save final model
        self.save_checkpoint(save_dir / 'final_model.pth')
        
        # Plot training curves
        self.plot_training_curves(save_dir)
    
    def save_checkpoint(self, path):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        print(f"Checkpoint loaded from {path}")
    
    def plot_training_curves(self, save_dir):
        """Plot training and validation loss curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_dir / 'training_curves.png')
        plt.close()
        print(f"Training curves saved to {save_dir / 'training_curves.png'}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='TEN VAD Trainer')
    parser.add_argument('--train_data', type=str, required=True, 
                       help='Directory containing training audio files')
    parser.add_argument('--val_data', type=str, required=True,
                       help='Directory containing validation audio files')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Save directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--max_seq_len', type=int, default=1000, help='Maximum sequence length')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Check if data directories exist
    if not os.path.exists(args.train_data):
        print(f"Error: Training data directory not found: {args.train_data}")
        sys.exit(1)
    
    if not os.path.exists(args.val_data):
        print(f"Error: Validation data directory not found: {args.val_data}")
        sys.exit(1)
    
    # Initialize feature extractor
    feature_extractor = VADFeatureExtractor()
    
    # Create datasets
    train_dataset = VADDataset(args.train_data, feature_extractor, args.max_seq_len)
    val_dataset = VADDataset(args.val_data, feature_extractor, args.max_seq_len)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    # Initialize model
    model = TenVAD()
    
    # Initialize trainer
    trainer = VADTrainer(model, device=args.device)
    trainer.setup_optimizer(learning_rate=args.lr, weight_decay=args.weight_decay)
    
    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        trainer.load_checkpoint(args.checkpoint)
        print(f"Resuming training from epoch {trainer.epoch + 1}")
    
    # Start training
    trainer.train(train_loader, val_loader, args.epochs, args.save_dir)
    
    print("Training completed!")


if __name__ == "__main__":
    main() 