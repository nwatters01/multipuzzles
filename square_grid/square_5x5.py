"""Square 5x5 puzzle."""

from matplotlib import pyplot as plt
import numpy as np

import sys
sys.path.append("../src")
import base_puzzle
import base_arrangement
import edge_wibble
from warping import stretch
from warping import twist
from pieces import square_piece

_RANDOM_SEED = 1


class Puzzle(base_puzzle.BasePuzzle):

    def __init__(self, width=5, height=5):
        """Constructor."""
        self._width = width
        self._height = height
        self._num_pieces = width * height
        
        # Create all the pieces
        pieces = [
            square_piece.SquarePiece(side_length=1, resolution=10, label=str(i))
            for i in range(self._num_pieces)
        ]
        
        # Create the puzzle
        super().__init__(pieces)
        
        # Add a random arrangement
        self.add_random_arrangement()
        self.add_random_arrangement()
        
        # Add warpings
        stretch_warping = stretch.Stretch(theta=np.pi / 4, stretch_factor=1.5)
        self.add_warping(stretch_warping, arrangement_index=0)
        twist_warping = twist.Twist(
            puzzle_bounds=np.array([[0, 0], [width, height]]),
            num_twists=5,
            rotation_magnitude=np.pi / 2,
            basin_size=3,
        )
        self.add_warping(twist_warping, arrangement_index=0)
        
        # Add edge wibbling
        sample_noise=edge_wibble.get_sample_noise()
        curvature_object = edge_wibble.Curvature(
            sample_curvature_magnitude=(
                edge_wibble.get_sample_curvature_magnitude()),
            sample_curve_length=edge_wibble.get_sample_curve_length(),
            sample_straight_length=edge_wibble.get_sample_straight_length(),
            sample_noise=sample_noise,
            resolution=500,
        )
        flattening_fn = edge_wibble.get_flattening_fn()
        wibbled_path_object = edge_wibble.WibbledPath(
            curvature_object, flattening_fn)
        self.wibble_edges(wibbled_path_object)
        
        # Update pixel positions and values
        for piece in self._pieces:
            piece.pixelate()
            
        # Save the puzzle
        # self.save("../logs/square_5x5_v0")
        
    def add_random_arrangement(self):
        """Add a random arrangement."""
        
        # Find target location for each piece
        target_positions = [
            (i, j) for i in range(self._height) for j in range(self._width)
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
        

if __name__ == "__main__":
    np.random.seed(_RANDOM_SEED)
    puzzle = Puzzle()
    puzzle.plot_arrangements()
    plt.show()
        