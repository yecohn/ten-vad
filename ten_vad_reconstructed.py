import torch
import torch.nn as nn
import torch.nn.functional as F

class TenVAD(nn.Module):
    def __init__(self):
        super().__init__()
        # ─── Conv front-end ──────────────────────────────────────────
        self.conv_dw = nn.Conv2d(
            1,
            1,
            kernel_size=(3, 3),
            bias=False,
        )
        self.conv_pw = nn.Conv2d(1, 16, kernel_size=1, bias=True)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d((1, 3), stride=(1, 2))

        # first 1D-separable conv, uses stride 2 along width just like the ONNX model
        self.sep1_dw = nn.Conv2d(
            16,
            16,
            kernel_size=(1, 3),
            stride=(2, 2),  # stride-2 in time/width dimension
            padding=(0, 1),
            groups=16,
            bias=False,
        )
        self.sep1_pw = nn.Conv2d(16, 16, kernel_size=1, bias=True)

        # second 1D-separable conv, again stride-2 along width
        self.sep2_dw = nn.Conv2d(
            16,
            16,
            kernel_size=(1, 3),
            stride=(2, 2),
            groups=16,
            bias=False,
        )
        self.sep2_pw = nn.Conv2d(16, 16, kernel_size=1, bias=True)

        # ─── RNN core ────────────────────────────────────────────────
        # After the conv stack we will build 5-frame windows so feature dim = 16*5 = 80
        self.window_size = 5  # equals context window used in original ONNX
        self.lstm1 = nn.LSTM(80, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 64, batch_first=True)

        # ─── Densities ───────────────────────────────────────────────
        self.fc1 = nn.Linear(128, 32)  # 128 → 32
        self.fc2 = nn.Linear(32, 1)  # 32  → 1
        self.sig = nn.Sigmoid()

    def forward(self, input_1, input_2=None, input_3=None, input_6=None, input_7=None):
        input_1 = input_1.unsqueeze(-1)  # Add channel dimension: (B, 3, 41) → (B, 1, 3, 41)
        input_1 = input_1.reshape(-1, 1, 3, 41)

        input_1 = self.conv_dw(input_1)  # (B, 1, 1, 41)
        input_1 = self.conv_pw(input_1)  # (B,16, 1, 41)
        input_1 = self.relu(input_1)
        input_1 = self.pool(input_1)  # (B,16, 1, 20)

        # ─ separable_conv1d (really 2-D) ─
        input_1 = self.sep1_dw(input_1)  # (B,16,1,20)
        input_1 = self.sep1_pw(input_1)
        input_1 = input_1.squeeze(2)  # (B,16,20)
        input_1 = self.relu(input_1)

        # put the singleton height back for the second block
        input_1 = input_1.unsqueeze(2)  # (B,16,1,20)
        input_1 = F.pad(input_1, (0,1,0,0), mode='constant', value=0)
        input_1 = self.sep2_dw(input_1)  # (B,16,1,20)
        input_1 = self.sep2_pw(input_1)
        input_1 = input_1.squeeze(2)  # (B,16,20)
        input_1 = self.relu(input_1)

        input_1 = input_1.permute(0, 2, 1)  # (B, width=5, 16)
        B, W, C = input_1.shape  # W should be 5 after the stride-2 convs
        input_1 = input_1.reshape(B, 1, W * C)  # (B, 1, 80)

        # LSTM stack
        input_2 = input_2.unsqueeze(0)
        input_3 = input_3.unsqueeze(0)
        input_6 = input_6.unsqueeze(0)
        input_7 = input_7.unsqueeze(0)

        x1, (input_2, input_3) = self.lstm1(input_1, (input_2, input_3))  # (B, T, 64)
        x2, (input_6, input_7) = self.lstm2(x1, (input_6, input_7))  # (B, T, 64)

        # ONNX model concatenates outputs of both LSTM layers along feature axis -> 128
        input_1 = torch.cat([x2, x1], dim=2)  # (B, T, 128)
        # dense layers
        input_1 = self.fc1(input_1)
        input_1 = self.relu(input_1)
        input_1 = self.fc2(input_1)
        return (self.sig(input_1), input_2, input_3, input_6, input_7)
