import torch
import torch.nn as nn


class TenVAD(nn.Module):
    def __init__(self):
        super().__init__()
        # ─── Conv front-end ──────────────────────────────────────────
        self.conv_dw = nn.Conv2d(
            1,
            1,
            kernel_size=(3, 3),
            padding=(0, 1),  # <── only width is padded
            bias=False,
        )
        self.conv_pw = nn.Conv2d(1, 16, kernel_size=1, bias=True)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d((1, 2))

        # keep the time-axis length by padding width with 1 on both sides
        self.sep1_dw = nn.Conv2d(
            16, 16, kernel_size=(1, 3), padding=(0, 1), groups=16, bias=False
        )
        self.sep1_pw = nn.Conv2d(16, 16, kernel_size=1, bias=True)

        self.sep2_dw = nn.Conv2d(
            16, 16, kernel_size=(1, 3), padding=(0, 1), groups=16, bias=False
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

    def forward(self, x, h1=None, c1=None, h2=None, c2=None):
        # x: (B, 3, 41)  → (B, 1, 3, 41)
        B = x.size(0)
        x = x.unsqueeze(1)

        x = self.conv_dw(x)  # (B, 1, 1, 41)
        x = self.conv_pw(x)  # (B,16, 1, 41)
        x = self.relu(x)
        x = self.pool(x)  # (B,16, 1, 20)

        # ─ separable_conv1d (really 2-D) ─
        x = self.sep1_dw(x)  # (B,16,1,20)
        x = self.sep1_pw(x)
        x = self.relu(x)
        x = x.squeeze(2)  # (B,16,20)

        # put the singleton height back for the second block
        x = x.unsqueeze(2)  # (B,16,1,20)
        x = self.sep2_dw(x)  # (B,16,1,20)
        x = self.sep2_pw(x)
        x = self.relu(x)
        x = x.squeeze(2)  # (B,16,20)

        # ─── build sliding windows of 5 frames to get feature dim 80 ───
        # result: (B, T=16, 16*5)
        x = (
            x.unfold(dimension=2, size=self.window_size, step=1)  # (B,16,16,5)
            .permute(0, 2, 1, 3)  # (B,16,16,5) -> (B,T,16,5)
            .reshape(x.size(0), -1, 16 * self.window_size)
        )

        # LSTM stack
        h1 = torch.zeros(1, B, 64, device=x.device) if h1 is None else h1
        c1 = torch.zeros(1, B, 64, device=x.device) if c1 is None else c1
        x1, (h1, c1) = self.lstm1(x, (h1, c1))  # (B, T, 64)

        h2 = torch.zeros(1, B, 64, device=x.device) if h2 is None else h2
        c2 = torch.zeros(1, B, 64, device=x.device) if c2 is None else c2
        x2, (h2, c2) = self.lstm2(x1, (h2, c2))  # (B, T, 64)

        # ONNX model concatenates outputs of both LSTM layers along feature axis -> 128
        x = torch.cat([x1, x2], dim=2)  # (B, T, 128)

        # dense layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.sig(x), (h1, c1, h2, c2)
