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
        self.sep2_dw = nn.Conv2d(
            16,
            16,
            kernel_size=(1, 3),
            stride=(2, 2),
            groups=16,
            bias=False,
        )
        self.sep2_pw = nn.Conv2d(16, 16, kernel_size=1, bias=True)
        self.window_size = 5  # equals context window used in original ONNX
        self.lstm1 = nn.LSTM(80, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 64, batch_first=True)
        self.fc1 = nn.Linear(128, 32)  # 128 → 32
        self.fc2 = nn.Linear(32, 1)  # 32  → 1
        self.sig = nn.Sigmoid()

    def forward(self, x, lstm1_h=None, lstm1_b=None, lstm2_h=None, lstm2_b=None):
        x = x.unsqueeze(-1)  
        x = x.reshape(-1, 1, 3, 41)

        x = self.conv_dw(x)  
        x = self.conv_pw(x)  
        x = self.relu(x)
        x = self.pool(x)  

        x = self.sep1_dw(x)  
        x = self.sep1_pw(x)
        x = x.squeeze(2) 
        x = self.relu(x)

        x = x.unsqueeze(2)  
        x = F.pad(x, (0,1,0,0), mode='constant', value=0)
        x = self.sep2_dw(x)  
        x = self.sep2_pw(x)
        x = x.squeeze(2)  
        x = self.relu(x)

        x = x.permute(0, 2, 1)  
        B, W, C = x.shape  
        x = x.reshape(B, 1, W * C)  

        # LSTM stack
        lstm1_h = lstm1_h.unsqueeze(0)
        lstm1_b = lstm1_b.unsqueeze(0)
        lstm2_h = lstm2_h.unsqueeze(0)
        lstm2_b = lstm2_b.unsqueeze(0)

        x1, (lstm1_h, lstm1_b) = self.lstm1(x, (lstm1_h, lstm1_b))  
        x2, (lstm2_h, lstm2_b) = self.lstm2(x1, (lstm2_h, lstm2_b))  

        x = torch.cat([x2, x1], dim=2)  
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return (self.sig(x), lstm1_h, lstm1_b, lstm2_h, lstm2_b)
