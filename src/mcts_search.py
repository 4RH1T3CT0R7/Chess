"""Monte Carlo Tree Search implementation (simplified)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass, field
from typing import Dict, Optional

import chess
import torch

from .nn_evaluator import NNEvaluator
if TYPE_CHECKING:
    from .core_engine import EngineConfig


@dataclass
class Node:
    """Basic tree node for MCTS."""

    board: chess.Board
    parent: Optional["Node"] = None
    children: Dict[chess.Move, "Node"] = field(default_factory=dict)
    visits: int = 0
    value: float = 0.0


class MCTSSearch:
    """Simplified MCTS wrapper that calls the evaluator."""

    def __init__(self, evaluator: NNEvaluator) -> None:
        self.evaluator = evaluator

    def reset(self) -> None:  # pragma: no cover - placeholder
        """Reset search state if needed."""

    def best_move(self, fen: str, config: "EngineConfig") -> str:
        """Return the best move string for given position."""
        board = chess.Board(fen)
        best_move: Optional[chess.Move] = None
        best_score = float("-inf")
        for move in board.legal_moves:
            board.push(move)
            tensor = self.board_to_tensor(board)
            score = self.evaluator.evaluate(tensor)
            score = self.adjust_for_humanity(score, config.humanity)
            if score > best_score:
                best_score = score
                best_move = move
            board.pop()
        return best_move.uci() if best_move else "0000"

    @staticmethod
    def board_to_tensor(board: chess.Board) -> torch.Tensor:
        """Convert board to tensor representation."""
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

    @staticmethod
    def adjust_for_humanity(score: float, humanity: int) -> float:
        """Apply a penalty to mimic human-like play."""
        factor = 1.0 - humanity / 20.0
        return score * factor
