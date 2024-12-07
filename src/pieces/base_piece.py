"""Base piece class."""

import numpy as np


class BasePiece:
    """Base piece class."""
    
    def __init__(self, vertices: np.array, label=None):
        """Constructor.
        
        Args:
            vertices (np.array): Array of shape (n, 2) containing the vertices
                in order such that each consecutive pair of vertices forms an
                edge, with wrapping around to the first point.
            label (str): Label for the piece.
        """
        self._vertices = vertices
        self._label = label
        self._num_sides = len(vertices)
        self._edges = []
        for i in range(self._num_sides):
            self._edges.append((vertices[i - 1], vertices[i]))
            
    def recenter(self):
        return translation
    
    @property
    def vertices(self) -> np.array:
        """Return the vertices."""
        return self._vertices

    @property
    def edges(self) -> list:
        """Return the edges."""
        return self._edges
    
    @property
    def num_sides(self) -> int:
        """Return the number of edges."""
        return self._num_sides
    
    @property
    def label(self) -> str:
        """Return the label."""
        return self._label