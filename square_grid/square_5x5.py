"""Square 5x5 puzzle."""

from matplotlib import pyplot as plt
import numpy as np

import sys
sys.path.append("../src")
import base_puzzle
import base_arrangement
from pieces import square_piece


class Puzzle(base_puzzle.BasePuzzle):

    def __init__(self):
        """Constructor."""
        
        # Create all the pieces
        pieces = [
            square_piece.SquarePiece(side_length=1)
            for _ in range(25)
        ]
        
        # Create the puzzle
        super().__init__(pieces)
        
        # Add one 5x5 grid arrangement
        transforms = [
            base_arrangement.Transform(
                translation=np.array([i, j]),
                theta=0.,
            )
            for i in range(5)
            for j in range(5)
        ]
        arrangement = base_arrangement.BaseArrangement(
            pieces=self._pieces, transforms=transforms
        )
        self.add_arrangement(arrangement)
        

if __name__ == "__main__":
    puzzle = Puzzle()
    puzzle.plot_arrangements()
    plt.show()
        