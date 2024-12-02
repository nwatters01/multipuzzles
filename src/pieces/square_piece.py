"""Square piece."""

from pieces import base_piece
import numpy as np


class SquarePiece(base_piece.BasePiece):
    """Square piece."""
    
    def __init__(self, side_length: float):
        """Constructor.
        
        Args:
            side_length (float): Side length of the square
        """
        vertices = side_length * 0.5 * np.array([
            [1, 1], [-1, 1], [-1, -1], [1, -1],
        ])
        super().__init__(vertices)