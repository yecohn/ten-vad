# TEN VAD Trainer

This directory contains a comprehensive training system for the TEN VAD (Voice Activity Detection) model, based on the `process_frame` method from the VAD demo.

## Overview

The training system consists of several components:

1. **VADTrainer** (`vad_trainer.py`) - Main training class with PyTorch integration
2. **VADFeatureExtractor** - Feature extraction using the same pipeline as the demo
3. **VADDataset** - PyTorch dataset for loading audio files and labels
4. **Data Preparation** (`prepare_vad_data.py`) - Tools for preparing training data
5. **Example Script** (`train_vad_example.py`) - Complete training example

## Features

- **Frame-by-frame training**: Uses the same `process_frame` approach as the demo
- **LSTM state management**: Properly handles hidden states during training
- **Feature extraction**: Identical to the demo's feature extraction pipeline
- **Flexible data loading**: Supports various audio formats and label formats
- **Synthetic data generation**: Can generate training data for testing
- **Checkpoint management**: Save/load training progress
- **Training visualization**: Plot training curves
- **Multi-GPU support**: Compatible with PyTorch's distributed training

## Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install numpy scipy librosa matplotlib tqdm
```

### 2. Run the Example

```bash
python train_vad_example.py
```

This will:
- Create synthetic audio files
- Generate VAD labels
- Train the model for 5 epochs
- Test inference on a sample file

### 3. Train with Your Own Data

```bash
# Prepare your dataset
python prepare_vad_data.py --input /path/to/audio/files --output vad_dataset --method energy

# Train the model
python vad_trainer.py --train_data vad_dataset/train --val_data vad_dataset/val --epochs 50
```

## Data Format

### Audio Files
- Supported formats: `.wav`, `.flac`, `.mp3`
- Sample rate: 16kHz (will be resampled if different)
- Duration: Any length (will be truncated to `max_seq_len` frames)

### Label Files
Labels should be in tab-separated format:
```
Start(s)	End(s)	Duration(s)	Label
0.000	2.500	2.500	speech
2.500	4.000	1.500	silence
4.000	6.500	2.500	speech
```

If no label file exists, the system will generate synthetic labels using one of these methods:
- `energy`: Based on audio energy threshold
- `random`: Random alternating segments
- `librosa`: Using librosa's voice activity detection

## Training Configuration

### Command Line Arguments

```bash
python vad_trainer.py \
    --train_data /path/to/train/data \
    --val_data /path/to/val/data \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --save_dir checkpoints \
    --device cuda \
    --max_seq_len 1000
```

### Key Parameters

- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--lr`: Learning rate
- `--weight_decay`: Weight decay for regularization
- `--max_seq_len`: Maximum sequence length (frames)
- `--device`: Device to use ('cuda' or 'cpu')

## Model Architecture

The trainer uses the `TenVAD` model which consists of:

1. **Convolutional Frontend**: 2D convolutions for feature extraction
2. **LSTM Layers**: Two bidirectional LSTM layers
3. **Dense Layers**: Final classification layers
4. **Sigmoid Output**: VAD probability [0, 1]

## Training Process

### Frame-by-Frame Processing

The trainer processes audio frame-by-frame, just like the demo:

1. **Feature Extraction**: Extract mel-frequency features from each 16ms frame
2. **Context Window**: Create 3-frame context windows
3. **LSTM Processing**: Pass through LSTM layers with state management
4. **Loss Computation**: Binary cross-entropy loss for each frame
5. **Backpropagation**: Update model parameters

### State Management

- Hidden states are reset for each sequence
- States are properly passed between frames
- Compatible with the ONNX export format

## Output Files

### Checkpoints
- `best_model.pth`: Best model based on validation loss
- `final_model.pth`: Final model after training
- `checkpoint_epoch_N.pth`: Checkpoints every 10 epochs

### Training Curves
- `training_curves.png`: Plot of training and validation loss

## Advanced Usage

### Custom Loss Functions

You can modify the loss function in `VADTrainer`:

```python
# Example: Focal loss for imbalanced data
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

trainer.criterion = FocalLoss()
```

### Custom Data Loading

You can create custom datasets by inheriting from `VADDataset`:

```python
class CustomVADDataset(VADDataset):
    def _load_labels(self, audio_file):
        # Your custom label loading logic
        pass
```

### Multi-GPU Training

```python
# Wrap model for multi-GPU
model = torch.nn.DataParallel(model)
trainer = VADTrainer(model, device='cuda')
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `batch_size` or `max_seq_len`
2. **Slow Training**: Use GPU, increase `num_workers` in DataLoader
3. **Poor Convergence**: Adjust learning rate, check data quality
4. **Label Mismatch**: Ensure label files match audio files

### Debug Mode

Add debug prints to the training loop:

```python
# In VADTrainer.forward_pass
print(f"Input shape: {features.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Prediction: {prediction.item():.4f}")
```

## Integration with ONNX

The trained model can be exported to ONNX:

```python
# Load trained model
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Export to ONNX
torch.onnx.export(model, dummy_input, "trained_vad.onnx")
```

## Performance Tips

1. **Data Preprocessing**: Pre-extract features for faster training
2. **Mixed Precision**: Use `torch.cuda.amp` for faster training
3. **Gradient Accumulation**: For larger effective batch sizes
4. **Early Stopping**: Implement early stopping based on validation loss

## Example Training Session

```bash
# 1. Prepare data
python prepare_vad_data.py --input audio_files --output dataset --method librosa

# 2. Train model
python vad_trainer.py \
    --train_data dataset/train \
    --val_data dataset/val \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-3 \
    --save_dir my_checkpoints

# 3. Monitor training
# Check training_curves.png for progress
# Monitor checkpoints/best_model.pth for best model

# 4. Test inference
python examples_onnx/vad_demo.py \
    --input test_audio.wav \
    --model my_checkpoints/best_model.pth
```

## File Structure

```
.
├── vad_trainer.py          # Main training class
├── prepare_vad_data.py     # Data preparation utilities
├── train_vad_example.py    # Complete training example
├── ten_vad_reconstructed.py # PyTorch model definition
├── examples_onnx/
│   └── vad_demo.py         # Original demo (for reference)
└── README_VAD_TRAINER.md   # This file
```

## License

This code is part of the TEN Framework and is licensed under the Apache License, Version 2.0. 