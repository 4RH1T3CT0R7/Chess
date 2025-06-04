# Chess Assistant (Research Prototype)

This repository contains a simplified implementation of a chess engine
assistant designed for research into human–AI interaction. It consists of
four core modules located in `src/`:

- `core_engine.py` – engine manager and factory
- `nn_evaluator.py` – residual CNN evaluator

- `nn_evaluator.py` – minimal PyTorch evaluator

- `mcts_search.py` – MCTS search with optional "humanity" factor
- `gui_app.py` – cross-platform GUI using PySimpleGUI

Helper stubs for potential online integration live in `src/helpers/` and
are disabled by default. A small self-play generator script demonstrates
basic usage.
