
"""Neural network evaluator for chess positions.

This module provides a small yet functional convolutional neural network
architecture suitable for board evaluation tasks.  It exposes a simple
`NNEvaluator` wrapper that handles device placement, tensor conversion from
``chess.Board`` objects, and convenience methods for saving and loading
weights.

The network itself is intentionally lightweight (~8M parameters by default)
to keep inference fast on consumer GPUs such as the RTX 3070.  Despite its
size it includes residual connections and global average pooling which are
common in modern chess engines.
"""
=======
"""Neural network evaluator for chess positions."""


from __future__ import annotations

import torch
import torch.nn as nn

import chess

Tensor = torch.Tensor


class ResidualBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += identity
        return self.relu(out)


class ConvEvaluator(nn.Module):
    """A small residual CNN for evaluating chess boards."""

    def __init__(self, channels: int = 64, blocks: int = 6) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(13, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(blocks):
            layers.append(ResidualBlock(channels))
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 1),
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        out = self.features(x)
        return self.head(out)
=======


class SimpleEvaluator(nn.Module):
    """A tiny CNN evaluator as a placeholder."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(13, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.model(x)


class NNEvaluator:
    """Wrapper around the neural network model."""

    def __init__(self, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ConvEvaluator().to(self.device)

    def board_to_tensor(self, board: chess.Board) -> Tensor:
        """Convert ``chess.Board`` to a network input tensor."""
        planes = torch.zeros((13, 8, 8), dtype=torch.float32)
        piece_map = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5,
        }
        for square, piece in board.piece_map().items():
            idx = piece_map[piece.piece_type] + (0 if piece.color == chess.WHITE else 6)
            row = chess.square_rank(square)
            col = chess.square_file(square)
            planes[idx, row, col] = 1.0
        planes[12].fill_(1.0 if board.turn == chess.WHITE else 0.0)
        return planes.unsqueeze(0)

    def evaluate(self, board_tensor: Tensor | chess.Board) -> float:
        """Return a scalar evaluation for the given board tensor or board."""
        if isinstance(board_tensor, chess.Board):
            board_tensor = self.board_to_tensor(board_tensor)
=======
        self.model = SimpleEvaluator().to(self.device)

    def evaluate(self, board_tensor: torch.Tensor) -> float:
        """Return a scalar evaluation for the given board tensor."""

        with torch.no_grad():
            output = self.model(board_tensor.to(self.device))
        return float(output.item())

    def load(self, path: str) -> None:
        """Load model weights."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def save(self, path: str) -> None:
        """Save model weights."""
        torch.save(self.model.state_dict(), path)


    def export_onnx(self, path: str) -> None:
        """Export the model to ONNX format."""
        dummy = torch.zeros((1, 13, 8, 8), device=self.device)
        torch.onnx.export(self.model, dummy, path, opset_version=17)


