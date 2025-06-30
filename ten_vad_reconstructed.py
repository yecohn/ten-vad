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
            padding=(0, 1),  # asymmetric pad in ONNX, (0,0,0,1); (0,1) is close enough
            groups=16,
            bias=False,
        )
        self.sep2_pw = nn.Conv2d(16, 16, kernel_size=1, bias=True)

        # ─── RNN core ────────────────────────────────────────────────
        # After the conv stack we will build 5-frame windows so feature dim = 16*5 = 80
        self.window_size = 5  # equals context window used in original ONNX
        self.lstm1 = nn.LSTM(80, 64)
        self.lstm2 = nn.LSTM(64, 64)

        # ─── Densities ───────────────────────────────────────────────
        self.fc1 = nn.Linear(128, 32)  # 128 → 32
        self.fc2 = nn.Linear(32, 1)  # 32  → 1
        self.sig = nn.Sigmoid()

    def forward(self, input_1, input_2=None, input_3=None, input_6=None, input_7=None):
        # x: (B, 3, 41)  → (B, 1, 3, 41)
        # B = input_1.size(0)
        input_1 = input_1.unsqueeze(1)

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
        input_1 = self.sep2_dw(input_1)  # (B,16,1,20)
        input_1 = self.sep2_pw(input_1)
        input_1 = input_1.squeeze(2)  # (B,16,20)
        input_1 = self.relu(input_1)

        # ─── match the ONNX reshape sequence ───
        # ONNX steps: transpose (0,2,1) then reshape to (-1,1,80)
        # Here we keep batch_first so final tensor is (B, 1, 80)
        input_1 = input_1.permute(0, 2, 1)  # (B, width=5, 16)
        B, W, C = input_1.shape  # W should be 5 after the stride-2 convs
        input_1 = input_1.reshape(B, 1, W * C)  # (B, 1, 80)

        # LSTM stack
        #put this in input output 
        # h1 = torch.zeros(1, B, 64, device=x.device) if h1 is None else h1
        # c1 = torch.zeros(1, B, 64, device=x.device) if c1 is None else c1
        input_2 = input_2.unsqueeze(0)
        input_3 = input_3.unsqueeze(0)
        input_6 = input_6.unsqueeze(0)
        input_7 = input_7.unsqueeze(0)

        x1, (input_2, input_3) = self.lstm1(input_1, (input_2, input_3))  # (B, T, 64)
        # input_2 = input_2.squeeze(0)
        # input_3 = input_3.squeeze(0)

        # h2 = torch.zeros(1, B, 64, device=x.device) if h2 is None else h2
        # c2 = torch.zeros(1, B, 64, device=x.device) if c2 is None else c2
        x2, (input_6, input_7) = self.lstm2(x1, (input_6, input_7))  # (B, T, 64)

        # ONNX model concatenates outputs of both LSTM layers along feature axis -> 128
        input_1 = torch.cat([x2, x1], dim=2)  # (B, T, 128)
        # shape_input1 = torch.tensor(input_1.shape, dtype=torch.int32)
        # shape_input1_gather = torch.gather(shape_input1, 0, torch.tensor([0, 1]))
        # shape_input1_gather = torch.cat([shape_input1_gather, torch.tensor([32])], axis=0).to(torch.int64)
        # input_1 = input_1.reshape(shape_input1_gather)

        # dense layers
        input_1 = self.fc1(input_1)
        input_1 = self.relu(input_1)
        input_1 = self.fc2(input_1)
        # return (self.sig(input_1), input_2, input_3, input_6, input_7)
        return (self.sig(input_1).cpu().numpy(), input_2.cpu().numpy().squeeze(0), input_3.cpu().numpy().squeeze(0), input_6.cpu().numpy().squeeze(0), input_7.cpu().numpy().squeeze(0))
