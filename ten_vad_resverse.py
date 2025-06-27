import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnxruntime as ort
import numpy as np
from collections import OrderedDict

# --- Model Parameters (from previous refinement) ---
RAW_CONTEXT_WINDOW_LEN = 3
RAW_FEA_LEN = 41
GRU_INPUT_SIZE = 80
GRU_HIDDEN_DIM = 64
NUM_GRU_LAYERS = 4
DENSE_1_OUT_DIM = 32
INTERMEDIATE_DENSE_OUTPUT_DIM = 128
FINAL_OUTPUT_DIM = 1

class TenVAD(nn.Module):
    """
    PyTorch implementation of the VAD model, reconstructed from ONNX graph insights
    and accompanying C code.
    """
    def __init__(self, raw_input_seq_len, raw_input_fea_len, gru_input_size,
                 gru_hidden_dim, num_gru_layers, final_output_dim=1,
                 intermediate_dense_out_dim=128, dense_1_out_dim=32):
        super(TenVAD, self).__init__()

        self.raw_input_seq_len = raw_input_seq_len
        self.raw_input_fea_len = raw_input_fea_len
        self.gru_hidden_dim = gru_hidden_dim
        self.num_gru_layers = num_gru_layers

        # Feature Extraction Block (Convolutional part before GRU)
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(raw_input_fea_len, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(64, gru_input_size, kernel_size=1),
            nn.ReLU()
        )

        # Stacked GRU layers
        self.gru = nn.GRU(gru_input_size, gru_hidden_dim, num_gru_layers, batch_first=True)

        # Post-GRU Linear Layers
        self.intermediate_dense = nn.Linear(gru_hidden_dim, intermediate_dense_out_dim)
        self.dense_3 = nn.Linear(intermediate_dense_out_dim, dense_1_out_dim)
        self.dense_5 = nn.Linear(dense_1_out_dim, final_output_dim)

    def forward(self, x_raw_features, hidden_states):
        x_raw_features = x_raw_features.float()

        # Feature Extraction
        x_conv_input = x_raw_features.permute(0, 2, 1)
        x_conv_output = self.feature_extractor(x_conv_input)
        x_gru_input = x_conv_output.permute(0, 2, 1)

        h0 = torch.cat(hidden_states, dim=0)
        
        output_gru, hn = self.gru(x_gru_input, h0)

        last_timestep_output = output_gru[:, -1, :]

        # Post-GRU Linear Layers
        x = F.relu(self.intermediate_dense(last_timestep_output))
        x = F.relu(self.dense_3(x))
        vad_logits = self.dense_5(x)

        output_prob = torch.sigmoid(vad_logits)
        
        # Ensure output probability matches ONNX shape (1, 1, 1)
        # Squeeze if it's (1, 1) from sigmoid, then unsqueeze to (1, 1, 1)
        if output_prob.dim() == 2 and output_prob.shape[1] == 1:
            output_prob = output_prob.unsqueeze(2) # Add a dimension at the end

        next_hidden_states = [hn[i].unsqueeze(0) for i in range(self.num_gru_layers)]

        return output_prob, next_hidden_states


def load_onnx_weights_into_pytorch(pytorch_model, onnx_model_path):
    """
    Loads weights from an ONNX model into a PyTorch model.
    Only attempts to load weights for identifiable layers (Linear layers).
    """
    onnx_model = onnx.load(onnx_model_path)
    onnx_initializers = {init.name: onnx.numpy_helper.to_array(init) for init in onnx_model.graph.initializer}

    pytorch_state_dict = pytorch_model.state_dict()
    loaded_count = 0

    print("\n--- Attempting to load ONNX weights into PyTorch model ---")

    # Actual ONNX weights are for dense_3 and dense_5
    onnx_to_pytorch_map = {
        'StatefulPartitionedCall/vad_model/dense_3/Tensordot/ReadVariableOp:0': 'dense_3.weight',
        'StatefulPartitionedCall/vad_model/dense_3/BiasAdd/ReadVariableOp:0': 'dense_3.bias',
        'StatefulPartitionedCall/vad_model/dense_5/Tensordot/ReadVariableOp:0': 'dense_5.weight',
        'StatefulPartitionedCall/vad_model/dense_5/BiasAdd/ReadVariableOp:0': 'dense_5.bias',
    }

    for onnx_name, pytorch_name in onnx_to_pytorch_map.items():
        if onnx_name in onnx_initializers and pytorch_name in pytorch_state_dict:
            # Make ONNX array writable to avoid UserWarning
            onnx_param = np.asarray(onnx_initializers[onnx_name]).copy() 
            
            # PyTorch Linear layers expect weights to be (out_features, in_features)
            # TensorFlow/ONNX often store them as (in_features, out_features)
            # Check shape and transpose if necessary
            if pytorch_name.endswith('.weight') and len(onnx_param.shape) == 2:
                if pytorch_state_dict[pytorch_name].shape == onnx_param.T.shape:
                    pytorch_state_dict[pytorch_name].copy_(torch.from_numpy(onnx_param.T))
                    print(f"Loaded {onnx_name} (transposed) -> {pytorch_name}")
                    loaded_count += 1
                else:
                    print(f"Shape mismatch for {onnx_name} -> {pytorch_name}: PyTorch {pytorch_state_dict[pytorch_name].shape}, ONNX {onnx_param.shape}")
            else:
                if pytorch_state_dict[pytorch_name].shape == onnx_param.shape:
                    pytorch_state_dict[pytorch_name].copy_(torch.from_numpy(onnx_param))
                    print(f"Loaded {onnx_name} -> {pytorch_name}")
                    loaded_count += 1
                else:
                    print(f"Shape mismatch for {onnx_name} -> {pytorch_name}: PyTorch {pytorch_state_dict[pytorch_name].shape}, ONNX {onnx_param.shape}")
        else:
            print(f"Warning: ONNX initializer '{onnx_name}' or PyTorch parameter '{pytorch_name}' not found for loading.")

    # Apply the updated state dict to the model
    pytorch_model.load_state_dict(pytorch_state_dict, strict=False) # strict=False to allow partial loading
    
    # Warnings for layers that cannot be directly loaded
    print("\n--- Warnings for un-loaded components ---")
    print("WARNING: GRU weights (W0, R0, B0) cannot be directly loaded due to dimension mismatches or complex ONNX GRU operator structure.")
    print("         The ONNX GRU's internal dimension (256) does not align with standard PyTorch GRU(hidden_size=64).")
    print("WARNING: Feature extractor (Conv1d layers) weights cannot be directly loaded due to complex separable convolution mapping.")
    print("         These layers in the PyTorch model will retain their random initializations.")
    print(f"Total parameters loaded: {loaded_count}")


def compare_models(pytorch_model, onnx_file_path):
    """
    Compares the outputs of the PyTorch model and the ONNX model.
    """
    # 1. Load ONNX model with ONNX Runtime
    session = ort.InferenceSession(onnx_file_path)
    onnx_input_names = [inp.name for inp in session.get_inputs()]
    onnx_output_names = [out.name for out in session.get_outputs()]

    print(f"\nONNX Runtime will use inputs: {onnx_input_names}")
    print(f"ONNX Runtime will produce outputs: {onnx_output_names}")

    # 2. Prepare identical dummy inputs
    # Raw features input for ONNX and PyTorch
    dummy_input_features_np = np.random.randn(1, RAW_CONTEXT_WINDOW_LEN, RAW_FEA_LEN).astype(np.float32)
    dummy_input_features_torch = torch.from_numpy(dummy_input_features_np)

    # Hidden states for ONNX and PyTorch
    # C code's input_data_buf_1234 each {1, AUP_AED_MODEL_HIDDEN_DIM}
    # ONNX inputs are input_2, input_3, input_6, input_7 (each [0, 64])
    dummy_initial_hidden_states_np = [
        np.zeros((1, GRU_HIDDEN_DIM), dtype=np.float32) for _ in range(NUM_GRU_LAYERS)
    ]
    dummy_initial_hidden_states_torch = [
        torch.from_numpy(h).unsqueeze(0) for h in dummy_initial_hidden_states_np
    ]

    # Create ONNX input dict
    onnx_inputs = {
        onnx_input_names[0]: dummy_input_features_np, # input_1
        onnx_input_names[1]: dummy_initial_hidden_states_np[0], # input_2
        onnx_input_names[2]: dummy_initial_hidden_states_np[1], # input_3
        onnx_input_names[3]: dummy_initial_hidden_states_np[2], # input_6
        onnx_input_names[4]: dummy_initial_hidden_states_np[3], # input_7
    }

    # 3. Run inference on ONNX model
    print("\n--- Running ONNX Model Inference ---")
    onnx_outputs_raw = session.run(onnx_output_names, onnx_inputs)
    onnx_output_prob = onnx_outputs_raw[0] # output_1
    onnx_new_hidden_states = onnx_outputs_raw[1:] # output_2, output_3, output_6, output_7

    print(f"ONNX Output Probability Shape: {onnx_output_prob.shape}")
    for i, h_state in enumerate(onnx_new_hidden_states):
        print(f"ONNX New Hidden State {i+1} Shape: {h_state.shape}")

    # 4. Run inference on PyTorch model
    print("\n--- Running PyTorch Model Inference ---")
    pytorch_output_prob, pytorch_new_hidden_states_list = pytorch_model(
        dummy_input_features_torch, dummy_initial_hidden_states_torch
    )
    pytorch_new_hidden_states_np = [h.squeeze(0).cpu().detach().numpy() for h in pytorch_new_hidden_states_list] # Convert to numpy for comparison

    print(f"PyTorch Output Probability Shape: {pytorch_output_prob.shape}")
    for i, h_state in enumerate(pytorch_new_hidden_states_np):
        print(f"PyTorch New Hidden State {i+1} Shape: {h_state.shape}")

    # 5. Compare outputs
    print("\n--- Comparing Outputs ---")
    tolerance = 1e-4 # Epsilon for floating point comparison

    # Compare VAD Probability
    prob_diff = np.abs(onnx_output_prob - pytorch_output_prob.detach().numpy()).max()
    print(f"Max absolute difference in VAD Probability: {prob_diff}")
    if prob_diff < tolerance:
        print("VAD Probability outputs are numerically close (within tolerance).")
    else:
        print("VAD Probability outputs are NOT numerically close.")
        print("This is expected if GRU/feature_extractor weights were not loaded.")

    # Compare Hidden States
    for i in range(NUM_GRU_LAYERS):
        h_diff = np.abs(onnx_new_hidden_states[i] - pytorch_new_hidden_states_np[i]).max()
        print(f"Max absolute difference in Hidden State {i+1}: {h_diff}")
        if h_diff < tolerance:
            print(f"Hidden State {i+1} outputs are numerically close (within tolerance).")
        else:
            print(f"Hidden State {i+1} outputs are NOT numerically close.")
            print("This is expected if GRU/feature_extractor weights were not loaded.")


if __name__ == "__main__":
    onnx_file = "src/onnx_model/ten-vad.onnx" # Make sure this path is correct

    # Instantiate PyTorch model
    pytorch_vad_model = TenVAD(
        raw_input_seq_len=RAW_CONTEXT_WINDOW_LEN,
        raw_input_fea_len=RAW_FEA_LEN,
        gru_input_size=GRU_INPUT_SIZE,
        gru_hidden_dim=GRU_HIDDEN_DIM,
        num_gru_layers=NUM_GRU_LAYERS
    )
    print("PyTorch Model Architecture:")
    print(pytorch_vad_model)

    # Load weights from ONNX into PyTorch model (partial loading)
    load_onnx_weights_into_pytorch(pytorch_vad_model, onnx_file)

    # Set PyTorch model to evaluation mode (important for consistent behavior of layers like BatchNorm, Dropout if they were present)
    pytorch_vad_model.eval()

    # Compare models
    compare_models(pytorch_vad_model, onnx_file)