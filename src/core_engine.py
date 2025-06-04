"""Core engine module for the Chess Assistant.

This module coordinates the search algorithms and neural network
components. It also serves as the entry point for command line or
GUI-based sessions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .nn_evaluator import NNEvaluator
from .mcts_search import MCTSSearch


@dataclass
class EngineConfig:
    """Configuration parameters for the chess engine."""

    humanity: int = 5  # 0 (strongest) .. 10 (most human)
    max_depth: int = 16
    use_gpu: bool = True


def create_default_engine() -> "ChessEngine":
    """Factory that creates a ChessEngine with default settings."""
    evaluator = NNEvaluator()
    search = MCTSSearch(evaluator)
    return ChessEngine(config=EngineConfig(), evaluator=evaluator, search=search)


class ChessEngine:
    """Main engine controlling the search and evaluator."""

    def __init__(self, config: EngineConfig, evaluator: NNEvaluator, search: MCTSSearch) -> None:
        self.config = config
        self.evaluator = evaluator
        self.search = search

    def suggest_move(self, fen: str) -> str:
        """Return the best move for the given board position in FEN."""
        return self.search.best_move(fen, self.config)

    def load_weights(self, path: str) -> None:
        """Load neural network weights from disk."""
        self.evaluator.load(path)

    def save_weights(self, path: str) -> None:
        """Export neural network weights to disk."""
        self.evaluator.save(path)

    def reset(self) -> None:
        """Reset any internal state if needed."""
        self.search.reset()
