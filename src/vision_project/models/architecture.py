"""Model Architecture Definition.

Contains the PyTorch CRNN (Convolutional Recurrent Neural Network) architecture
tailored for sequence recognition (e.g., text / OCR) and CTC loss calculation.

6GB VRAM Optimization:
----------------------
- Total parameters: ~8.3M (~33MB in FP32).
- Micro-batch memory footprint during forward pass is < 150MB with AMP.
- Fits comfortably inside 6GB VRAM GPUs with high batch sizes or gradient accumulation.
"""

from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNN(nn.Module):
    """Convolutional Recurrent Neural Network for text and sequence recognition.

    Takes a grayscale image tensor of shape (B, 1, 32, W) and extracts features
    via 7 CNN layers, passes them through a 2-layer Bidirectional LSTM, and
    projects to class log probabilities for CTC loss calculation.
    """

    def __init__(
        self,
        img_h: int = 32,
        in_channels: int = 1,
        num_classes: int = 37,
        hidden_size: int = 256,
        dropout: float = 0.0,
    ) -> None:
        """Initialize the CRNN.

        Args:
            img_h: Expected input image height (must be 32).
            in_channels: Input channels (1 for grayscale).
            num_classes: Number of target characters including blank token (blank at index 0).
            hidden_size: Hidden dimension of each LSTM direction (output will be hidden_size * 2).
            dropout: Dropout probability between recurrent layers.
        """
        super().__init__()
        assert img_h == 32, f"Input image height must be exactly 32, got {img_h}"
        self.num_classes = num_classes

        # ---------------------------------------------------------------------
        # Feature Extraction: 7 CNN Layers
        # ---------------------------------------------------------------------
        # Layer 1: Conv(1, 64) -> ReLU -> MaxPool(2, 2)  [H: 32 -> 16]
        # Layer 2: Conv(64, 128) -> ReLU -> MaxPool(2, 2) [H: 16 -> 8]
        # Layer 3: Conv(128, 256) -> BatchNorm -> ReLU
        # Layer 4: Conv(256, 256) -> ReLU -> MaxPool((2,2), stride=(2,1), pad=(0,1)) [H: 8 -> 4]
        # Layer 5: Conv(256, 512) -> BatchNorm -> ReLU
        # Layer 6: Conv(512, 512) -> ReLU -> MaxPool((2,2), stride=(2,1), pad=(0,1)) [H: 4 -> 2]
        # Layer 7: Conv(512, 512, k=2, s=1, p=0) -> BatchNorm -> ReLU [H: 2 -> 1]
        self.cnn = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Layer 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Layer 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Layer 4
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)),
            # Layer 5
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # Layer 6
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)),
            # Layer 7
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # ---------------------------------------------------------------------
        # Sequence Modeling: 2-Layer Bidirectional LSTM
        # ---------------------------------------------------------------------
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=False,
            dropout=dropout if dropout > 0.0 else 0.0,
        )

        # Linear projection to class output
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for CRNN.

        Args:
            x: Input grayscale image tensor of shape (B, 1, 32, W).

        Returns:
            Log-probabilities tensor of shape (T, B, num_classes) suitable for nn.CTCLoss.
        """
        # Feature extraction through 7 CNN layers: (B, 1, 32, W) -> (B, 512, 1, W')
        features = self.cnn(x)

        # Squeeze height dimension: (B, 512, 1, W') -> (B, 512, W')
        features = features.squeeze(2)

        # Permute to (Time, Batch, Channels) for PyTorch LSTM: (W', B, 512)
        features = features.permute(2, 0, 1)

        # Recurrent sequence modeling: (T, B, hidden_size * 2)
        recurrent, _ = self.rnn(features)

        # Linear projection to vocabulary: (T, B, num_classes)
        logits = self.fc(recurrent)

        # Compute log softmax across the class dimension for CTC Loss
        log_probs = F.log_softmax(logits, dim=2)
        return log_probs
