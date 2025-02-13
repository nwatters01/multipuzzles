"""Base piece class."""

from matplotlib import path as matplotlib_path
import numpy as np
from pathlib import Path


class BasePiece:
    """Base piece class."""
    
    def __init__(self, vertices: np.array, resolution: int = 100, label=None):
        """Constructor.
        
        Args:
            vertices (np.array): Array of shape (n, 2) containing the vertices
                in order such that each consecutive pair of vertices forms an
                edge, with wrapping around to the first point.
            resolution (int): Number of pixels to use for pixelation.
            label (str): Label for the piece.
        """
        self._vertices = vertices
        self._resolution = resolution
        self._label = label
        self._num_sides = len(vertices)
        self.vertex_indices_per_edge = [
            ((i - 1) % self._num_sides, i) for i in range(self._num_sides)
        ]
        self.edges = self.get_edges()
        self.pixelate()
    
    def get_edges(self):
        edges = []
        for i in range(self._num_sides):
            edges.append((self._vertices[i - 1], self._vertices[i]))
        return edges
    
    def pixelate(self):
        """Set pixel positions and values."""
        # Find the bounding box of the piece's edges
        edges_array = np.concatenate([np.array(edge) for edge in self.edges])
        min_x, min_y = np.min(edges_array, axis=0)
        max_x, max_y = np.max(edges_array, axis=0)
        
        # Make a grid of pixel positions with given resolution
        x = np.arange(min_x, max_x, 1 / self._resolution)
        y = np.arange(min_y, max_y, 1 / self._resolution)
        pixel_positions = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
        
        # Remove pixel positions that are outside the piece
        edge_path = matplotlib_path.Path(edges_array)
        inside = edge_path.contains_points(pixel_positions)
        pixel_positions = pixel_positions[inside]
        
        # Create array of RGB pixel values, initialized as a random color
        random_color = np.random.rand(3)
        pixel_values = np.full((len(pixel_positions), 3), random_color)
        
        # Set pixel positions and values
        self.pixel_positions = pixel_positions
        self.pixel_values = pixel_values
        
    def save(self, path):
        """Save the piece attributes."""
        log_dir = Path(path)
        log_dir.mkdir(parents=True, exist_ok=True)
        np.save(log_dir / "vertices.npy", self._vertices)
        np.save(log_dir / "edges.npy", self.edges)
        np.save(log_dir / "pixel_positions.npy", self.pixel_positions)
        np.save(log_dir / "pixel_values.npy", self.pixel_values)
        if self._label is not None:
            with open(log_dir / "label.txt", "w") as f:
                f.write(self._label)
                
    @classmethod
    def load(cls, path):
        """Load the piece attributes."""
        log_dir = Path(path)
        vertices = np.load(log_dir / "vertices.npy")
        edges = np.load(log_dir / "edges.npy")
        pixel_positions = np.load(log_dir / "pixel_positions.npy")
        pixel_values = np.load(log_dir / "pixel_values.npy")
        label = None
        if (log_dir / "label.txt").exists():
            with open(log_dir / "label.txt", "r") as f:
                label = f.read()
        
        # Create piece and override edges and pixels
        piece = cls(vertices=vertices, label=label)
        piece.edges = edges
        piece.pixel_positions = pixel_positions
        piece.pixel_values = pixel_values
        
        return piece
       
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