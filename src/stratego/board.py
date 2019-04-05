import logging
from typing import Union
from pathlib import Path


class Board:
    SEP = "|"

    class ImporterKeys(Enum):
        board_dim = "BoardDim"

    def __init__(self):
        self.n_rows = self.n_colors = None

    def num_rows(self) -> int:
        r""" Accessor for number of rows on the board """
        return self._rows

    def num_cols(self) -> int:
        r""" Accessor for number of columns on the board """
        return self._cols

    @staticmethod
    def importer(file_path: Union[Path, str]) -> 'Board':
        r"""
        Board generator that follows the Factory Method pattern.

        :param file_path: Path to a board file.
        :return: Board with parameters in in \p file_path
        """
        if isinstance(file_path, str): file_path = Path(file_path)

        try:
            with open(file_path, "r") as f_in:
                lines = f_in.readlines()
        except IOError:
            raise ValueError("Unable to read board file.")

        brd = Board()
        for line in lines:
            spl = line.split(Board.SEP)
            if spl[0] == Board.ImporterKeys.board_dim.value:
                assert len(spl) == 3 and brd._rows is None and brd._cols is None
                brd._rows, brd._cols = int(spl[1]), int(spl[2])

    def is_valid(self):
        if self._rows is None or self._cols is None or self.num_rows <= 0 or self.num_cols <= 0:
            logging.warning("Invalid number of rows/column in board.")
            return False

        return True

    @staticmethod
    def _print_template_file(file_path: Union[Path, str]):
        lines = []
        lines.append([Board.ImporterKeys.board_dim.value, "num_rows", "num_cols"])

        file_path = Path(file_path)
        file_path.parent.mkdir(exist_ok=True)
        with open(file_path, "w+") as f_out:
            f_write("\n".join([Board.sep.join(l) for l in lines])
        assert False
