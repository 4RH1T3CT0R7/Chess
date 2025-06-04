"""Core engine module for the Chess Assistant.

This module coordinates the search algorithms and neural network
components. It also serves as the entry point for command line or
GUI-based sessions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import chess
import torch

from .nn_evaluator import NNEvaluator
from .mcts_search import MCTSSearch


class AlphaBetaSearch:
    """Basic alpha-beta search using the evaluator."""

    def __init__(self, evaluator: NNEvaluator) -> None:
        self.evaluator = evaluator

    def best_move(self, board: chess.Board, depth: int, config: "EngineConfig") -> str:
        best_val = float("-inf")
        best_move: Optional[chess.Move] = None
        for move in board.legal_moves:
            board.push(move)
            val = -self._search(board, depth - 1, float("-inf"), float("inf"), -1, config)
            board.pop()
            if val > best_val:
                best_val = val
                best_move = move
        return best_move.uci() if best_move else "0000"

    def _search(self, board: chess.Board, depth: int, alpha: float, beta: float, color: int, config: "EngineConfig") -> float:
        if depth == 0 or board.is_game_over():
            return color * self._evaluate(board, config)
        for move in board.legal_moves:
            board.push(move)
            val = -self._search(board, depth - 1, -beta, -alpha, -color, config)
            board.pop()
            if val > alpha:
                alpha = val
            if alpha >= beta:
                break
        return alpha

    def _evaluate(self, board: chess.Board, config: "EngineConfig") -> float:
        tensor = self._board_to_tensor(board)
        score = self.evaluator.evaluate(tensor)
        return self._adjust_for_humanity(score, config.humanity)

    @staticmethod
    def _adjust_for_humanity(score: float, humanity: int) -> float:
        factor = 1.0 - humanity / 20.0
        return score * factor

    @staticmethod
    def _board_to_tensor(board: chess.Board) -> torch.Tensor:
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


@dataclass
class EngineConfig:
    """Configuration parameters for the chess engine."""

    humanity: int = 5  # 0 (strongest) .. 10 (most human)
    max_depth: int = 3
    use_gpu: bool = True
    use_mcts: bool = True


def create_default_engine() -> "ChessEngine":
    """Factory that creates a ChessEngine with default settings."""
    evaluator = NNEvaluator()
    mcts = MCTSSearch(evaluator)
    ab = AlphaBetaSearch(evaluator)
    return ChessEngine(
        config=EngineConfig(), evaluator=evaluator, mcts=mcts, alpha_beta=ab
    )


class ChessEngine:
    """Main engine controlling the search and evaluator."""

    def __init__(
        self,
        config: EngineConfig,
        evaluator: NNEvaluator,
        mcts: MCTSSearch,
        alpha_beta: AlphaBetaSearch,
    ) -> None:
        self.config = config
        self.evaluator = evaluator
        self.mcts = mcts
        self.alpha_beta = alpha_beta

    def suggest_move(self, fen: str) -> str:
        """Return the best move for the given board position in FEN."""
        if self.config.use_mcts:
            return self.mcts.best_move(fen, self.config)
        board = chess.Board(fen)
        return self.alpha_beta.best_move(board, self.config.max_depth, self.config)

    def load_weights(self, path: str) -> None:
        """Load neural network weights from disk."""
        self.evaluator.load(path)

    def save_weights(self, path: str) -> None:
        """Export neural network weights to disk."""
        self.evaluator.save(path)

    def reset(self) -> None:
        """Reset any internal state if needed."""
        self.mcts.reset()
