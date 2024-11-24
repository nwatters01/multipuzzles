"""Square piece."""

import base_piece
import numpy as np


class Square(base_piece.BasePiece):
    """Square piece."""
    
    def __init__(self, side_length: float):
        """Constructor.
        
        Args:
            side_length (float): Side length of the square
        """
        radius = side_length / np.sqrt(2)
        vertices = np.array([
            [radius + np.sin(i * np.pi / 2), radius - np.cos(i *np.pi / 2)]
            for i in range(4)
        ])
        super().__init__(vertices)