import re

from .location import Location
from .move import Move
from .agent import Agent
from .player import Player


class HumanAgent(Agent):
    def __init__(self, plyr: Player):
        super().__init__(plyr)

    def get_next_move(self) -> Move:
        r""" Gets a valid move uniformly at random """
        while True:
            print("Player %s - Enter you next move in the form:")
            print("   (CurrentRow,CurrentColumn) (NewRow,NewColumn)")
            m_str = input("Move: ").strip()

            # Verify the input format is correct
            if not re.match(r"\(\d+,\s*\d+\)\s*\(\d+,\s*\d+\)", m_str):
                print("\nInvalid input. Try again\n")
                continue

            try:
                locs = [int(x) for x in re.findall(r"\d+", m_str)]
                orig, new = Location(*locs[:2]), Location(*locs[2:])
            except ValueError:
                print("\nIt appears you entered an invalid location. Try again\n")
                continue

            piece = self._plyr.get_piece_at_loc(orig)
            if piece is None:
                print("\nIt appears you do not have a piece at the specified location. Try again\n")
                continue

            move = self._plyr.get_move(piece, new)
            if move is None:
                msg = ("\nIt does not appear you are able to move your piece to that location. "
                       "Try again\n")
                print(msg)
                continue

            return move
