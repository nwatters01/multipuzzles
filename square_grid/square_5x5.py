"""Square 5x5 puzzle."""

from matplotlib import pyplot as plt
import numpy as np

import sys
sys.path.append("../src")
import base_puzzle
import base_arrangement
from pieces import square_piece

_RANDOM_SEED = 0


class Puzzle(base_puzzle.BasePuzzle):

    def __init__(self, width=5, height=5):
        """Constructor."""
        self._width = width
        self._height = height
        self._num_pieces = width * height
        
        # Create all the pieces
        pieces = [
            square_piece.SquarePiece(side_length=1, label=str(i))
            for i in range(self._num_pieces)
        ]
        
        # Create the puzzle
        super().__init__(pieces)
        
        # Add a random arrangement
        self.add_random_arrangement()
        self.add_random_arrangement()
        
    def add_random_arrangement(self):
        """Add a random arrangement."""
        
        # Find target location for each piece
        target_locations = [
            (i, j) for i in range(self._height) for j in range(self._width)
        ]
        np.random.shuffle(target_locations)
        
        # Create transforms
        transforms = []
        for piece_index in range(self._num_pieces):
            theta = np.random.choice([i * np.pi / 2 for i in range(4)])
            transform = base_arrangement.Transform(
                translation=np.array(target_locations[piece_index]),
                theta=theta,
            )
            transforms.append(transform)
            
        # Create arrangement
        arrangement = base_arrangement.BaseArrangement(
            pieces=self._pieces, transforms=transforms
        )
        self.add_arrangement(arrangement)
        

if __name__ == "__main__":
    np.random.seed(_RANDOM_SEED)
    puzzle = Puzzle()
    puzzle.plot_arrangements()
    plt.show()
        