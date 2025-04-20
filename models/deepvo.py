# models/deepvo.py
import torch
import torch.nn as nn
from models.custom_cnn import CustomCNN

class DeepVO(nn.Module):
    """
    DeepVO network combining a CNN encoder and an LSTM to predict pose sequences.
    """
    def __init__(self, feature_dim=512, rnn_hidden=1000):
        super(DeepVO, self).__init__()
        # CNN backbone now expects 6-channel input
        self.encoder     = CustomCNN(in_channels=6, feature_dim=feature_dim)
        # Collapse spatial dimensions to 1×1
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # LSTM for temporal modeling
        self.rnn         = nn.LSTM(feature_dim, rnn_hidden, batch_first=True)
        # Regressor outputs 6-dim pose (3 Euler angles + 3 translations)
        self.regressor   = nn.Linear(rnn_hidden, 6)

    def forward(self, x_seq):
        """
        Args:
            x_seq: Tensor of shape (B, T, 6, H, W)
        Returns:
            preds: Tensor of shape (B, T, 6)
        """
        B, T, C, H, W = x_seq.shape
        feats = []
        for t in range(T):
            frame = x_seq[:, t]                # (B, 6, H, W)
            feat  = self.encoder(frame)        # (B, feature_dim, H', W')
            pooled = self.global_pool(feat)     # (B, feature_dim, 1, 1)
            vec    = pooled.view(B, -1)        # (B, feature_dim)
            feats.append(vec)
        seq_feats = torch.stack(feats, dim=1)  # (B, T, feature_dim)
        out, _    = self.rnn(seq_feats)        # (B, T, rnn_hidden)
        preds     = self.regressor(out)        # (B, T, 6)
        return preds
