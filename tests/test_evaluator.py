import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)+"/.."))
import chess
from src.nn_evaluator import NNEvaluator

def test_evaluate_board() -> None:
    board = chess.Board()
    evaluator = NNEvaluator(device="cpu")
    score = evaluator.evaluate(board)
    assert isinstance(score, float)
