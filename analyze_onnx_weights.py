#!/usr/bin/env python3
"""
Script to analyze ONNX model weights and their shapes
"""

import re
import numpy as np


def parse_onnx_initializer(raw_data, dims, data_type):
    """Parse ONNX initializer raw_data to numpy array"""
    # Handle different data types
    if data_type == 1:  # FLOAT
        dtype = np.float32
    elif data_type == 6:  # INT32
        dtype = np.int32
    elif data_type == 7:  # INT64
        dtype = np.int64
    else:
        raise ValueError(f"Unsupported data type: {data_type}")
    
    try:
        # Convert escaped string to bytes
        if isinstance(raw_data, str):
            # Remove quotes and handle octal escape sequences
            raw_data = raw_data.strip('"')
            
            # Convert octal escape sequences to bytes
            raw_bytes = b''
            i = 0
            while i < len(raw_data):
                if raw_data[i] == '\\' and i + 1 < len(raw_data):
                    if raw_data[i+1] == '\\':
                        raw_bytes += b'\\'
                        i += 2
                    elif raw_data[i+1] in '01234567' and i + 2 < len(raw_data):
                        # Octal escape sequence
                        octal_str = raw_data[i+1:i+4]
                        try:
                            byte_val = int(octal_str, 8)
                            raw_bytes += bytes([byte_val])
                            i += 4
                        except ValueError:
                            # Fallback to single character
                            raw_bytes += raw_data[i+1].encode('latin1')
                            i += 2
                    else:
                        # Other escape sequence
                        raw_bytes += raw_data[i+1].encode('latin1')
                        i += 2
                else:
                    raw_bytes += raw_data[i].encode('latin1')
                    i += 1
        else:
            raw_bytes = raw_data
        
        # Convert bytes to numpy array
        arr = np.frombuffer(raw_bytes, dtype=dtype)
        
        # Reshape according to dims
        if dims and len(dims) > 0:
            total_size = np.prod(dims)
            if arr.size >= total_size:
                arr = arr[:total_size].reshape(dims)
            else:
                print(f"Warning: Array size {arr.size} < expected size {total_size}")
        
        return arr
    except Exception as e:
        print(f"Error parsing raw_data: {e}")
        print(f"Raw data: {raw_data[:100]}...")
        print(f"Expected dims: {dims}")
        print(f"Data type: {data_type}")
        raise


def analyze_onnx_weights(onnx_file_path):
    """Analyze all weights in ONNX model"""
    with open(onnx_file_path, 'r') as f:
        content = f.read()
    
    initializer_pattern = r'initializer\s*\{([^}]+)\}'
    initializers = re.findall(initializer_pattern, content, re.DOTALL)
    
    print("ONNX Model Weight Analysis")
    print("=" * 50)
    
    weights_info = {}
    
    for i, init in enumerate(initializers):
        # Extract name
        name_match = re.search(r'name:\s*"([^"]+)"', init)
        if not name_match:
            continue
        name = name_match.group(1)
        
        # Extract dims
        dims = []
        dim_matches = re.findall(r'dims:\s*(\d+)', init)
        for dim in dim_matches:
            dims.append(int(dim))
        
        # Extract data type
        data_type_match = re.search(r'data_type:\s*(\d+)', init)
        if not data_type_match:
            continue
        data_type = int(data_type_match.group(1))
        
        # Extract raw_data
        raw_data_match = re.search(r'raw_data:\s*"([^"]+)"', init)
        if not raw_data_match:
            continue
        raw_data = raw_data_match.group(1)
        
        try:
            weight_array = parse_onnx_initializer(raw_data, dims, data_type)
            weights_info[name] = {
                'shape': weight_array.shape,
                'dtype': weight_array.dtype,
                'size': weight_array.size,
                'data_type': data_type
            }
            
            print(f"{i+1:2d}. {name}")
            print(f"     Shape: {weight_array.shape}")
            print(f"     Dtype: {weight_array.dtype}")
            print(f"     Size: {weight_array.size}")
            print()
            
        except Exception as e:
            print(f"Error parsing {name}: {e}")
            continue
    
    return weights_info


def suggest_pytorch_mapping(weights_info):
    """Suggest PyTorch state dict mapping based on weight shapes and names"""
    print("\nSuggested PyTorch State Dict Mapping")
    print("=" * 50)
    
    mapping = {}
    
    # Analyze by name patterns
    conv_weights = {k: v for k, v in weights_info.items() if 'conv' in k.lower()}
    lstm_weights = {k: v for k, v in weights_info.items() if any(x in k for x in ['W0__', 'R0__', 'B0__'])}
    dense_weights = {k: v for k, v in weights_info.items() if 'dense' in k.lower()}
    
    print("Convolution Weights:")
    for name, info in conv_weights.items():
        print(f"  {name} -> {suggest_conv_mapping(name, info)}")
    
    print("\nLSTM Weights:")
    for name, info in lstm_weights.items():
        print(f"  {name} -> {suggest_lstm_mapping(name, info)}")
    
    print("\nDense Weights:")
    for name, info in dense_weights.items():
        print(f"  {name} -> {suggest_dense_mapping(name, info)}")
    
    return mapping


def suggest_conv_mapping(name, info):
    """Suggest mapping for convolution weights"""
    if 'const_fold_opt__178' in name:
        return "conv_dw.weight"
    elif 'ReadVariableOp_1' in name and 'separable_conv2d' in name:
        return "conv_pw.weight"
    elif 'BiasAdd' in name and 'separable_conv2d' in name:
        return "conv_pw.bias"
    elif 'const_fold_opt__179' in name:
        return "sep1_dw.weight"
    elif 'ExpandDims_2' in name and 'separable_conv1d' in name and not '_1' in name:
        return "sep1_pw.weight"
    elif 'BiasAdd' in name and 'separable_conv1d' in name and not '_1' in name:
        return "sep1_pw.bias"
    elif 'const_fold_opt__180' in name:
        return "sep2_dw.weight"
    elif 'ExpandDims_2' in name and 'separable_conv1d_1' in name:
        return "sep2_pw.weight"
    elif 'BiasAdd' in name and 'separable_conv1d_1' in name:
        return "sep2_pw.bias"
    else:
        return f"Unknown conv: {name}"


def suggest_lstm_mapping(name, info):
    """Suggest mapping for LSTM weights"""
    if 'W0__70' in name:
        return "lstm1.weight_ih_l0 + lstm1.weight_hh_l0 (split)"
    elif 'R0__71' in name:
        return "lstm1.weight_hh_l0"
    elif 'B0__72' in name:
        return "lstm1.bias_ih_l0 + lstm1.bias_hh_l0 (split)"
    elif 'W0__99' in name:
        return "lstm2.weight_ih_l0 + lstm2.weight_hh_l0 (split)"
    elif 'R0__100' in name:
        return "lstm2.weight_hh_l0"
    elif 'B0__101' in name:
        return "lstm2.bias_ih_l0 + lstm2.bias_hh_l0 (split)"
    else:
        return f"Unknown LSTM: {name}"


def suggest_dense_mapping(name, info):
    """Suggest mapping for dense weights"""
    if 'Tensordot/ReadVariableOp' in name:
        if info['shape'][0] == 32:
            return "fc1.weight (transpose)"
        elif info['shape'][0] == 1:
            return "fc2.weight (transpose)"
        else:
            return f"Unknown dense weight: {name}"
    elif 'BiasAdd/ReadVariableOp' in name:
        if info['shape'][0] == 32:
            return "fc1.bias"
        elif info['shape'][0] == 1:
            return "fc2.bias"
        else:
            return f"Unknown dense bias: {name}"
    else:
        return f"Unknown dense: {name}"


def main():
    """Main analysis function"""
    weights_info = analyze_onnx_weights("onnx_model.txt")
    suggest_pytorch_mapping(weights_info)
    
    print(f"\nTotal weights found: {len(weights_info)}")
    
    # Save analysis to file
    with open("onnx_weights_analysis.txt", "w") as f:
        f.write("ONNX Weights Analysis\n")
        f.write("=" * 50 + "\n\n")
        for name, info in weights_info.items():
            f.write(f"{name}:\n")
            f.write(f"  Shape: {info['shape']}\n")
            f.write(f"  Dtype: {info['dtype']}\n")
            f.write(f"  Size: {info['size']}\n\n")
    
    print("Analysis saved to onnx_weights_analysis.txt")


if __name__ == "__main__":
    main() 