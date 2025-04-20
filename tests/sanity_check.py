# tests/sanity_check.py
import torch
from models.deepvo import DeepVO

model = DeepVO()
dummy = torch.randn(2, 5, 3, 184, 608)  # B=2, T=5, H=184, W=608
out = model(dummy)
print("Output shape:", out.shape)  # expect (2, 5, 15)
