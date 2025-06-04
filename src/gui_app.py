"""GUI Application for the Chess Assistant."""

from __future__ import annotations

from typing import List

import PySimpleGUI as sg
import chess
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .core_engine import create_default_engine


class ChessGUI:
    """Simple wrapper around PySimpleGUI."""

    def __init__(self) -> None:
        self.engine = create_default_engine()
        self.history: List[float] = []
        self.fig = Figure(figsize=(4, 2))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_ylabel("Eval")
        self.canvas_elem = sg.Canvas(key="CANVAS", size=(320, 160))

        self.layout = [
            [sg.Text("Mode"), sg.Combo(["Blitz", "Rapid", "Classical"], default_value="Blitz", key="MODE", readonly=True)],
            [sg.Text("Humanity"), sg.Slider(range=(0, 10), default_value=5, orientation="h", key="HUMANITY")],
            [sg.Text("Top N"), sg.Spin([1, 2, 3, 4, 5], initial_value=3, key="TOPN", size=(3, 1))],
            [sg.Multiline(size=(40, 4), key="FEN")],
            [sg.Button("Suggest", key="SUGGEST")],
            [sg.Multiline(size=(40, 5), key="OUTPUT", disabled=True)],
            [self.canvas_elem],
        ]
        self.window = sg.Window("Chess Assistant", self.layout, finalize=True)
        self.canvas = FigureCanvasTkAgg(self.fig, self.window["CANVAS"].TKCanvas)
        self.canvas.draw()

    def run(self) -> None:
        while True:
            event, values = self.window.read()
            if event == sg.WINDOW_CLOSED:
                break
            if event == "SUGGEST":
                fen = values["FEN"].strip()
                try:
                    chess.Board(fen)
                except Exception:
                    self.window["OUTPUT"].update("Invalid FEN")
                    continue
                self.engine.config.humanity = int(values["HUMANITY"])

                mode = values["MODE"]
                if mode == "Blitz":
                    self.engine.config.max_depth = 3
                    self.engine.config.mcts_iterations = 500
                elif mode == "Rapid":
                    self.engine.config.max_depth = 5
                    self.engine.config.mcts_iterations = 1000
                else:
                    self.engine.config.max_depth = 7
                    self.engine.config.mcts_iterations = 1500

                top_n = int(values["TOPN"])
                moves = self.engine.suggest_moves(fen, n=top_n)
                lines = [f"{i+1}. {m} ({s:.2f})" for i, (m, s) in enumerate(moves)]
                self.window["OUTPUT"].update("\n".join(lines))
                if moves:
                    self.history.append(moves[0][1])
                    self._update_plot()
        self.window.close()

    def _update_plot(self) -> None:
        self.ax.clear()
        self.ax.set_ylabel("Eval")
        self.ax.plot(self.history, marker="o")
        self.canvas.draw()


def main() -> None:
    gui = ChessGUI()
    gui.run()


if __name__ == "__main__":
    main()
