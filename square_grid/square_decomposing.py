"""Decomposing puzzle."""

from matplotlib import pyplot as plt
import numpy as np

import sys
sys.path.append("../src")
import base_puzzle
import base_arrangement
from warping import stretch
from warping import twist
from pieces import square_piece

_RANDOM_SEED = 1


class DecomposingPuzzle(base_puzzle.BasePuzzle):

    def __init__(self,
                 big_size=(9, 6),
                 decomposed_sizes=((3, 6), (3, 6), (3, 6))):
        """Constructor."""
        self._big_size = big_size
        self._decomposed_sizes = decomposed_sizes
        self._num_pieces = big_size[0] * big_size[1]
        
        # Sanity check that the sizes are consistent
        sum_decomposed_sizes = sum([
            decomposed_size[0] * decomposed_size[1]
            for decomposed_size in decomposed_sizes
        ])
        if sum_decomposed_sizes != self._num_pieces:
            raise ValueError(
                f"Decomposed sizes {sum_decomposed_sizes} do not match big "
                f"size {self._num_pieces}."
            )
        
        # Create all the pieces
        pieces = [
            square_piece.SquarePiece(side_length=1, label=str(i))
            for i in range(self._num_pieces)
        ]
        
        # Create the puzzle
        super().__init__(pieces)
        
        # Create the big arrangement
        self.add_big_arrangement()
        
        # Create the decomposed arrangements
        self.add_decomposed_arrangement()
        
        # Add warpings
        stretch_warping = stretch.Stretch(theta=np.pi / 4, stretch_factor=1.3)
        self.add_warping(stretch_warping, arrangement_index=0)
        twist_warping = twist.Twist(
            puzzle_bounds=np.array([[0, 0], list(self._big_size)]),
            num_twists=10,
            rotation_magnitude=0.6 * np.pi,
            basin_size=10,
        )
        self.add_warping(twist_warping, arrangement_index=0)
        
    def add_big_arrangement(self):
        """Add a random arrangement."""
        
        # Find target location for each piece
        target_positions = [
            (i, j)
            for i in range(self._big_size[0])
            for j in range(self._big_size[1])
        ]
        np.random.shuffle(target_positions)
        
        # Create transforms
        transforms = []
        for piece_index in range(self._num_pieces):
            theta = np.random.choice([i * np.pi / 2 for i in range(4)])
            translation = np.array(target_positions[piece_index], dtype=float)
            transform = base_arrangement.Transform(
                translation=translation,
                theta=theta,
            )
            transforms.append(transform)
            
        # Create arrangement
        arrangement = base_arrangement.BaseArrangement(
            pieces=self._pieces, transforms=transforms
        )
        self.add_arrangement(arrangement)
        
    def add_decomposed_arrangement(self, buffer_size=2):
        """Add decomposed arrangements."""
        
        # Target positions
        target_positions = []
        x_offset = 0
        for decomposed_size in self._decomposed_sizes:
            target_positions.extend([
                (x_offset + i, j)
                for i in range(decomposed_size[0])
                for j in range(decomposed_size[1])
            ])
            x_offset += decomposed_size[0] + buffer_size
            
        # Shuffle target positions
        np.random.shuffle(target_positions)
            
        # Get transform per target position
        transforms = []
        for piece_index in range(self._num_pieces):
            theta = np.random.choice([i * np.pi / 2 for i in range(4)])
            translation = np.array(target_positions[piece_index], dtype=float)
            transform = base_arrangement.Transform(
                translation=translation,
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
    puzzle = DecomposingPuzzle()
    puzzle.plot_arrangements()
    plt.show()
        