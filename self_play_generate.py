"""Self-play game generator (simplified)."""

from __future__ import annotations

import chess

from src.core_engine import create_default_engine


def generate_games(num_games: int = 10) -> None:
    engine = create_default_engine()
    for _ in range(num_games):
        board = chess.Board()
        while not board.is_game_over():
            move = engine.suggest_move(board.fen())
            board.push_uci(move)
        print(board.result())


if __name__ == "__main__":
    generate_games(1)
