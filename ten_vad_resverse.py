import onnx
import onnxruntime as rt
import numpy as np
import torch
from collections import OrderedDict

# TenVAD model -----------------------------------------------------
from ten_vad_cursor import TenVAD  # final architecture

# ------------------------------------------------------------------
# 1.  Read ONNX weights into a dict
# ------------------------------------------------------------------
m = onnx.load("src/onnx_model/ten-vad.onnx")
weights = {t.name: onnx.numpy_helper.to_array(t) for t in m.graph.initializer}


# ------------------------------------------------------------------
# 2.  Helper: convert ONNX → PyTorch for a single-direction LSTM
# ------------------------------------------------------------------
def onnx_lstm_to_torch(W, R, B):
    """Convert ONNX (TF-style) LSTM tensors to PyTorch layout.

    ONNX gate order: i o f c
    PyTorch gate order: i f g o   (g == candidate "c")
    """

    W = W.squeeze(0)
    R = R.squeeze(0)
    B = B.squeeze(0)

    # reorder gates ------------------------------------------------
    idx = [0, 2, 3, 1]  # i, f, c, o

    def reorder(mat):
        chunks = np.split(mat, 4, axis=0)
        return np.concatenate([chunks[i] for i in idx], axis=0)

    wb, rb = np.split(B, 2)
    return (
        torch.from_numpy(reorder(W)),
        torch.from_numpy(reorder(R)),
        torch.from_numpy(reorder(wb)),
        torch.from_numpy(reorder(rb)),
    )


# ------------------------------------------------------------------
# 3.  Build a new state_dict for TenVAD
# ------------------------------------------------------------------
state = OrderedDict()

# 3a. Conv front-end ----------------------------------------------
state["conv_dw.weight"] = torch.from_numpy(weights["const_fold_opt__178"])
state["conv_pw.weight"] = torch.from_numpy(
    weights[
        "StatefulPartitionedCall/vad_model/separable_conv2d/separable_conv2d/ReadVariableOp_1:0"
    ]
)
state["conv_pw.bias"] = torch.from_numpy(
    weights[
        "StatefulPartitionedCall/vad_model/separable_conv2d/BiasAdd/ReadVariableOp:0"
    ]
)

# 3b. Separable 1-D conv blocks -----------------------------------
state["sep1_dw.weight"] = torch.from_numpy(weights["const_fold_opt__179"])
state["sep1_pw.weight"] = torch.from_numpy(
    weights["StatefulPartitionedCall/vad_model/separable_conv1d/ExpandDims_2:0"]
)
state["sep1_pw.bias"] = torch.from_numpy(
    weights[
        "StatefulPartitionedCall/vad_model/separable_conv1d/BiasAdd/ReadVariableOp:0"
    ]
)

state["sep2_dw.weight"] = torch.from_numpy(weights["const_fold_opt__180"])
state["sep2_pw.weight"] = torch.from_numpy(
    weights["StatefulPartitionedCall/vad_model/separable_conv1d_1/ExpandDims_2:0"]
)
state["sep2_pw.bias"] = torch.from_numpy(
    weights[
        "StatefulPartitionedCall/vad_model/separable_conv1d_1/BiasAdd/ReadVariableOp:0"
    ]
)

# 3c. LSTM layers --------------------------------------------------
w_ih, w_hh, b_ih, b_hh = onnx_lstm_to_torch(
    weights["W0__70"], weights["R0__71"], weights["B0__72"]
)
state["lstm1.weight_ih_l0"] = w_ih
state["lstm1.weight_hh_l0"] = w_hh
state["lstm1.bias_ih_l0"] = b_ih
state["lstm1.bias_hh_l0"] = b_hh

w_ih, w_hh, b_ih, b_hh = onnx_lstm_to_torch(
    weights["W0__99"], weights["R0__100"], weights["B0__101"]
)
state["lstm2.weight_ih_l0"] = w_ih
state["lstm2.weight_hh_l0"] = w_hh
state["lstm2.bias_ih_l0"] = b_ih
state["lstm2.bias_hh_l0"] = b_hh

# 3d. Dense layers -------------------------------------------------
state["fc1.weight"] = torch.from_numpy(
    weights["StatefulPartitionedCall/vad_model/dense_3/Tensordot/ReadVariableOp:0"].T
)
state["fc1.bias"] = torch.from_numpy(
    weights["StatefulPartitionedCall/vad_model/dense_3/BiasAdd/ReadVariableOp:0"]
)

state["fc2.weight"] = torch.from_numpy(
    weights["StatefulPartitionedCall/vad_model/dense_5/Tensordot/ReadVariableOp:0"].T
)
state["fc2.bias"] = torch.from_numpy(
    weights["StatefulPartitionedCall/vad_model/dense_5/BiasAdd/ReadVariableOp:0"]
)


# ------------------------------------------------------------------
# 4.  Create model, load weights, validate -------------------------
# ------------------------------------------------------------------
def main():
    model = TenVAD()
    model.eval()

    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys: {unexpected}")
    print("✓  PyTorch model successfully initialised with ONNX weights")

    sess = rt.InferenceSession(
        "src/onnx_model/ten-vad.onnx", providers=["CPUExecutionProvider"]
    )

    # random test batch ------------------------------------------
    B = 1  # ONNX model expects batch size 1 for hidden states
    x_np = np.random.randn(B, 3, 41).astype(np.float32)
    zeros = np.zeros((B, 64), dtype=np.float32)

    onnx_out = sess.run(
        None,
        {
            "input_1": x_np,
            "input_2": zeros,
            "input_3": zeros,
            "input_6": zeros,
            "input_7": zeros,
        },
    )[0]

    with torch.no_grad():
        torch_out, _ = model(
            torch.from_numpy(x_np),
            torch.from_numpy(zeros).unsqueeze(0),
            torch.from_numpy(zeros).unsqueeze(0),
            torch.from_numpy(zeros).unsqueeze(0),
            torch.from_numpy(zeros).unsqueeze(0),
        )

    err = np.max(np.abs(torch_out.numpy() - onnx_out))
    print(f"max |onnx - torch| = {err:.4e}")


if __name__ == "__main__":
    main()
