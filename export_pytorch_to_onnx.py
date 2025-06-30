#!/usr/bin/env python3
"""
Script to export the loaded PyTorch model to ONNX format
"""

import torch
import torch.onnx
from ten_vad_reconstructed import TenVAD


def export_to_onnx(model, input_shape, output_path):
    """Export PyTorch model to ONNX format"""
    print(f"Exporting model to ONNX: {output_path}")
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dummy input tensors
    # Based on your model's forward method, you need:
    # input_1: (B, 3, 41) - main input
    # input_2: (B, 64) - lstm1 hidden state
    # input_3: (B, 64) - lstm1 cell state  
    # input_6: (B, 64) - lstm2 hidden state
    # input_7: (B, 64) - lstm2 cell state
    
    batch_size = 1
    input_1 = torch.randn(batch_size, 3, 41)
    input_2 = torch.randn(batch_size, 64)  # lstm1 hidden state
    input_3 = torch.randn(batch_size, 64)  # lstm1 cell state
    input_6 = torch.randn(batch_size, 64)  # lstm2 hidden state
    input_7 = torch.randn(batch_size, 64)  # lstm2 cell state
    
    # Create dummy inputs tuple
    dummy_inputs = (input_1, input_2, input_3, input_6, input_7)
    
    print("Input shapes:")
    print(f"  input_1: {input_1.shape}")
    print(f"  input_2: {input_2.shape}")
    print(f"  input_3: {input_3.shape}")
    print(f"  input_6: {input_6.shape}")
    print(f"  input_7: {input_7.shape}")
    
    # Export to ONNX
    torch.onnx.export(
        model,                     # model being run
        dummy_inputs,             # model input (or a tuple for multiple inputs)
        output_path,              # where to save the model
        export_params=True,       # store the trained parameter weights inside the model file
        opset_version=11,         # the ONNX version to export the model to
        do_constant_folding=True, # whether to execute constant folding for optimization
        input_names=['input_1', 'input_2', 'input_3', 'input_6', 'input_7'],  # the model's input names
        output_names=['output'],  # the model's output names
        dynamic_axes={
            'input_1': {0: 'batch_size'},
            'input_2': {0: 'batch_size'},
            'input_3': {0: 'batch_size'},
            'input_6': {0: 'batch_size'},
            'input_7': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported successfully to: {output_path}")


def main():
    """Main export function"""
    # Load the model
    print("Loading PyTorch model...")
    model = TenVAD()
    
    # Load the weights
    try:
        state_dict = torch.load("ten_vad_onnx_weights.pth")
        model.load_state_dict(state_dict, strict=True)
        print("Successfully loaded weights")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return
    
    # Export to ONNX
    output_path = "ten_vad_exported.onnx"
    export_to_onnx(model, (1, 3, 41), output_path)
    
    # Verify the exported model
    print("\nVerifying exported ONNX model...")
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model is valid!")
        
        # Print model info
        print(f"Model IR version: {onnx_model.ir_version}")
        print(f"Opset version: {onnx_model.opset_import[0].version}")
        print(f"Producer: {onnx_model.producer_name}")
        
        # Print input/output info
        print("\nModel inputs:")
        for input_info in onnx_model.graph.input:
            print(f"  {input_info.name}: {[dim.dim_value for dim in input_info.type.tensor_type.shape.dim]}")
        
        print("\nModel outputs:")
        for output_info in onnx_model.graph.output:
            print(f"  {output_info.name}: {[dim.dim_value for dim in output_info.type.tensor_type.shape.dim]}")
            
    except Exception as e:
        print(f"Error verifying ONNX model: {e}")


if __name__ == "__main__":
    main() 