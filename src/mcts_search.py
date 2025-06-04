
"""Monte Carlo Tree Search implementation.

This module provides a lightweight MCTS algorithm that consults the neural
network evaluator for leaf evaluation. It supports a configurable number of
iterations and applies a humanity adjustment so the engine can mimic more
human-like play when desired.
"""
=======
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
    """Tree node used by the MCTS algorithm."""
=======
    """Basic tree node for MCTS."""

    board: chess.Board
    parent: Optional["Node"] = None
    children: Dict[chess.Move, "Node"] = field(default_factory=dict)
    visits: int = 0
    value: float = 0.0


class MCTSSearch:
    """Monte Carlo Tree Search using the evaluator for rollouts."""
=======
    """Simplified MCTS wrapper that calls the evaluator."""

    def __init__(self, evaluator: NNEvaluator) -> None:
        self.evaluator = evaluator

    def reset(self) -> None:  # pragma: no cover - placeholder
        """Reset search state if needed."""

    def best_move(self, fen: str, config: "EngineConfig") -> str:
        """Return the best move for the position using MCTS."""
        root = Node(board=chess.Board(fen))

        for _ in range(config.mcts_iterations):
            node = self._select(root)
            value = self._simulate(node.board, config)
            self._backpropagate(node, value)

        if not root.children:
            return "0000"
        best_child = max(root.children.items(), key=lambda item: item[1].visits)[0]
        return best_child.uci()

    def _select(self, node: Node) -> Node:
        from math import log, sqrt

        while node.children and not node.board.is_game_over():
            total = node.visits
            node = max(
                node.children.values(),
                key=lambda child: self._ucb(child, total),
            )
        if node.visits == 0 or node.board.is_game_over():
            return node
        return self._expand(node)

    def _expand(self, node: Node) -> Node:
        for move in node.board.legal_moves:
            if move not in node.children:
                board = node.board.copy()
                board.push(move)
                child = Node(board=board, parent=node)
                node.children[move] = child
                return child
        # Should not reach here if called correctly
        return node

    def _simulate(self, board: chess.Board, config: "EngineConfig") -> float:
        tensor = self.board_to_tensor(board)
        score = self.evaluator.evaluate(tensor)
        score = self.adjust_for_humanity(score, config.humanity)
        return score if board.turn == chess.WHITE else -score

    def _backpropagate(self, node: Node, value: float) -> None:
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent
            value = -value

    @staticmethod
    def _ucb(child: Node, total_visits: int) -> float:
        from math import log, sqrt

        if child.visits == 0:
            return float("inf")
        c = 1.4
        return child.value / child.visits + c * sqrt(log(total_visits) / child.visits)
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
