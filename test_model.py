import torch
from ten_vad_reconstructed import TenVAD

# Test the PyTorch model
model = TenVAD()
model.eval()
print(model)

# Create test input
x = torch.randn(1, 3, 41)  # batch=1, 3 frames, 41 features

# Run inference
with torch.no_grad():
    output, states = model(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Output values: {output.squeeze()}")
print("✓ PyTorch model runs successfully!")

# Save the model
torch.save(model.state_dict(), "ten_vad_pytorch.pth")
print("✓ Model saved to ten_vad_pytorch.pth")
