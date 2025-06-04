"""GUI Application for the Chess Assistant."""

from __future__ import annotations

import PySimpleGUI as sg
import chess

from .core_engine import create_default_engine


class ChessGUI:
    """Simple wrapper around PySimpleGUI."""

    def __init__(self) -> None:
        self.engine = create_default_engine()
        self.layout = [
            [sg.Text("Humanity"), sg.Slider(range=(0, 10), default_value=5, orientation="h", key="HUMANITY")],
            [sg.Button("Suggest Move", key="SUGGEST")],
            [sg.Multiline(size=(40, 10), key="FEN")],
            [sg.Text("Suggested"), sg.Text(size=(10, 1), key="OUTPUT")],
        ]
        self.window = sg.Window("Chess Assistant", self.layout)

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
                move = self.engine.suggest_move(fen)
                self.window["OUTPUT"].update(move)
        self.window.close()


def main() -> None:
    gui = ChessGUI()
    gui.run()


if __name__ == "__main__":
    main()
