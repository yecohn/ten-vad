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
import argparse
import os
import sys
from pathlib import Path
import random
import librosa

# Constants
SAMPLE_RATE = 16000
HOP_SIZE = 256  # 16ms per frame


def generate_synthetic_vad_labels(audio_file, output_file, method='energy'):
    """
    Generate synthetic VAD labels for an audio file
    
    Args:
        audio_file: Path to audio file
        output_file: Path to output label file
        method: Method to generate labels ('energy', 'random', 'librosa')
    """
    # Load audio
    sample_rate, audio_data = wavfile.read(audio_file)
    
    # Convert to float32
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32)
    elif audio_data.dtype == np.int32:
        audio_data = audio_data.astype(np.float32) / 65536.0
    
    # Resample if necessary
    if sample_rate != SAMPLE_RATE:
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
        sample_rate = SAMPLE_RATE
    
    duration = len(audio_data) / sample_rate
    frame_duration = HOP_SIZE / SAMPLE_RATE
    
    if method == 'energy':
        labels = _generate_energy_based_labels(audio_data, frame_duration)
    elif method == 'random':
        labels = _generate_random_labels(duration, frame_duration)
    elif method == 'librosa':
        labels = _generate_librosa_labels(audio_data, sample_rate, frame_duration)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Save labels
    with open(output_file, 'w') as f:
        f.write("Start(s)\tEnd(s)\tDuration(s)\tLabel\n")
        for start_time, end_time, label in labels:
            duration_seg = end_time - start_time
            label_str = 'speech' if label == 1 else 'silence'
            f.write(f"{start_time:.3f}\t{end_time:.3f}\t{duration_seg:.3f}\t{label_str}\n")
    
    print(f"Generated labels for {audio_file} -> {output_file}")


def _generate_energy_based_labels(audio_data, frame_duration):
    """Generate labels based on audio energy"""
    num_frames = len(audio_data) // HOP_SIZE
    labels = []
    
    # Calculate energy for each frame
    energies = []
    for i in range(num_frames):
        start_idx = i * HOP_SIZE
        end_idx = start_idx + HOP_SIZE
        frame = audio_data[start_idx:end_idx]
        energy = np.mean(frame ** 2)
        energies.append(energy)
    
    # Normalize energies
    energies = np.array(energies)
    energy_threshold = np.percentile(energies, 30)  # 30th percentile as threshold
    
    # Generate segments
    current_label = 1 if energies[0] > energy_threshold else 0
    current_start = 0.0
    
    for i in range(1, len(energies)):
        frame_time = i * frame_duration
        new_label = 1 if energies[i] > energy_threshold else 0
        
        if new_label != current_label:
            # End current segment
            labels.append((current_start, frame_time, current_label))
            current_start = frame_time
            current_label = new_label
    
    # Add final segment
    labels.append((current_start, len(energies) * frame_duration, current_label))
    
    return labels


def _generate_random_labels(duration, frame_duration):
    """Generate random labels"""
    labels = []
    current_time = 0.0
    current_label = random.choice([0, 1])
    
    while current_time < duration:
        # Random segment duration between 0.5 and 3 seconds
        segment_duration = random.uniform(0.5, 3.0)
        end_time = min(current_time + segment_duration, duration)
        
        labels.append((current_time, end_time, current_label))
        current_time = end_time
        current_label = 1 - current_label  # Toggle label
    
    return labels


def _generate_librosa_labels(audio_data, sample_rate, frame_duration):
    """Generate labels using librosa voice activity detection"""
    # Use librosa's voice activity detection
    voice_intervals = librosa.effects.split(audio_data, top_db=20, frame_length=2048, hop_length=512)
    
    labels = []
    duration = len(audio_data) / sample_rate
    
    # Convert librosa intervals to our format
    current_time = 0.0
    
    for start_sample, end_sample in voice_intervals:
        start_time = start_sample / sample_rate
        end_time = end_sample / sample_rate
        
        # Add silence before speech if needed
        if start_time > current_time:
            labels.append((current_time, start_time, 0))
        
        # Add speech segment
        labels.append((start_time, end_time, 1))
        current_time = end_time
    
    # Add final silence if needed
    if current_time < duration:
        labels.append((current_time, duration, 0))
    
    return labels


def prepare_dataset(input_dir, output_dir, method='energy', train_split=0.8):
    """
    Prepare a complete dataset for VAD training
    
    Args:
        input_dir: Directory containing audio files
        output_dir: Output directory for organized dataset
        method: Method to generate labels
        train_split: Fraction of data to use for training
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    train_dir = output_path / 'train'
    val_dir = output_path / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all audio files
    audio_files = list(input_path.glob("*.wav"))
    audio_files.extend(list(input_path.glob("*.flac")))
    audio_files.extend(list(input_path.glob("*.mp3")))
    
    if not audio_files:
        print(f"No audio files found in {input_dir}")
        return
    
    # Shuffle files
    random.shuffle(audio_files)
    
    # Split into train/val
    split_idx = int(len(audio_files) * train_split)
    train_files = audio_files[:split_idx]
    val_files = audio_files[split_idx:]
    
    print(f"Found {len(audio_files)} audio files")
    print(f"Training: {len(train_files)} files")
    print(f"Validation: {len(val_files)} files")
    
    # Process training files
    print("\nProcessing training files...")
    for audio_file in train_files:
        output_file = train_dir / f"{audio_file.stem}.txt"
        generate_synthetic_vad_labels(audio_file, output_file, method)
        
        # Copy audio file
        import shutil
        shutil.copy2(audio_file, train_dir / audio_file.name)
    
    # Process validation files
    print("\nProcessing validation files...")
    for audio_file in val_files:
        output_file = val_dir / f"{audio_file.stem}.txt"
        generate_synthetic_vad_labels(audio_file, output_file, method)
        
        # Copy audio file
        import shutil
        shutil.copy2(audio_file, val_dir / audio_file.name)
    
    print(f"\nDataset prepared successfully!")
    print(f"Training data: {train_dir}")
    print(f"Validation data: {val_dir}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Prepare VAD training data')
    parser.add_argument('--input', type=str, required=True,
                       help='Input audio file or directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output file or directory')
    parser.add_argument('--method', type=str, default='energy',
                       choices=['energy', 'random', 'librosa'],
                       help='Method to generate VAD labels')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Fraction of data for training (when processing directory)')
    
    args = parser.parse_args()
    
    if os.path.isfile(args.input):
        # Process single file
        generate_synthetic_vad_labels(args.input, args.output, args.method)
    elif os.path.isdir(args.input):
        # Process directory
        prepare_dataset(args.input, args.output, args.method, args.train_split)
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        sys.exit(1)


if __name__ == "__main__":
    main() 