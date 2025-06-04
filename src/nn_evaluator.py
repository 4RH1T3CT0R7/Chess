"""Neural network evaluator for chess positions."""

from __future__ import annotations

import torch
import torch.nn as nn


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
