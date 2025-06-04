import sys, os; sys.path.append(os.path.abspath(os.path.dirname(__file__)+"/.."))
import importlib

MODULES = [
    "src.core_engine",
    "src.nn_evaluator",
    "src.mcts_search",
    "src.gui_app",
    "self_play_generate",
]


def test_imports() -> None:
    for mod in MODULES:
        assert importlib.import_module(mod)
