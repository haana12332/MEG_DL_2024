import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

class MEGNet(nn.Module):
    def __init__(self, num_classes, seq_len, in_channels, num_subjects, subject_emb_dim=10, cnn_hid_dim=128, rnn_hid_dim=64):
        super(MEGNet, self).__init__()
        self.subject_embedding = nn.Embedding(num_subjects, subject_emb_dim)
        
        self.cnn_blocks = nn.Sequential(
            ConvBlock(in_channels + subject_emb_dim, cnn_hid_dim),
            ConvBlock(cnn_hid_dim, cnn_hid_dim)
        )
        
        self.rnn = nn.LSTM(cnn_hid_dim, rnn_hid_dim, batch_first=True, bidirectional=True)
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(rnn_hid_dim * 2, num_classes)
        )

    def forward(self, X, subject_idxs):
        subject_features = self.subject_embedding(subject_idxs)
        subject_features = subject_features.unsqueeze(2).expand(-1, -1, X.size(2))
        
        X = torch.cat([X, subject_features], dim=1)
        X = self.cnn_blocks(X)
        X = X.transpose(1, 2)  # LSTM expects input shape (batch, seq_len, features)
        X, _ = self.rnn(X)
        X = X.transpose(1, 2)  # Change back to (batch, features, seq_len) for pooling
        return self.head(X)

class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, p_drop=0.1):
        super(ConvBlock, self).__init__()
        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, X):
        X = F.gelu(self.batchnorm0(self.conv0(X)))
        X = F.gelu(self.batchnorm1(self.conv1(X)))
        return self.dropout(X)
