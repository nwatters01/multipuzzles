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
        self.vertex_indices_per_edge = [
            ((i - 1) % self._num_sides, i) for i in range(self._num_sides)
        ]
        self.edges = self.get_edges()
    
    def get_edges(self):
        edges = []
        for i in range(self._num_sides):
            edges.append((self._vertices[i - 1], self._vertices[i]))
        return edges
       
    @property
    def vertices(self) -> np.array:
        """Return the vertices."""
        return self._vertices
    
    @vertices.setter
    def vertices(self, vertices: np.array):
        """Set the vertices."""
        
        # Ensure all edge are composed of two verices for (start, end)
        for edge in self.edges:
            if len(edge) != 2:
                raise ValueError(
                    "Trying to move vertices after edge wibbling is prohibited."
                )
        
        # Set the vertices and update the edges
        self._vertices = vertices
        self.edges = self.get_edges()
    
    @property
    def num_sides(self) -> int:
        """Return the number of edges."""
        return self._num_sides
    
    @property
    def label(self) -> str:
        """Return the label."""
        return self._label