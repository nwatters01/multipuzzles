"""Base arrangement class."""

import numpy as np
from pieces import base_piece


class Transform:
    """Affine transform."""
    
    def __init__(self, translation: np.array, theta: float):
        """Constructor.
        
        Args:
            translation (np.array): Translation vector with 2 elements.
            theta (float): Rotation angle in radians.
        """
        self._translation = translation
        self._theta = theta
        self._rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ])
        self._inverse_rotation_matrix = np.array([
            [np.cos(-theta), -np.sin(-theta)],
            [np.sin(-theta), np.cos(-theta)],
        ])
    
    def apply(self, points: np.array) -> np.array:
        """Apply the transform to a set of points.
        
        Args:
            points (np.array): Array of shape (n, 2) containing the points.
            
        Returns:
            np.array: Array of shape (n, 2) containing the transformed points.
        """
        rotated_points = np.dot(points, self._rotation_matrix)
        translated_points = rotated_points + self._translation
        return translated_points
    
    def inverse(self, points: np.array) -> np.array:
        """Apply the inverse transform to a set of points.
        
        Args:
            points (np.array): Array of shape (n, 2) containing the points.
            
        Returns:
            np.array: Array of shape (n, 2) containing the transformed points.
        """
        translated_points = points - self._translation
        rotated_points = np.dot(
            translated_points, self._inverse_rotation_matrix)
        return rotated_points


class BaseArrangement:
    
    def __init__(self,
                 pieces: list[base_piece.BasePiece],
                 transforms: list[Transform]):
        """Constructor.
        
        Args:
            pieces (list[base_piece.BasePiece]): List of pieces.
            transforms (list[Transform]): List of transforms.
        """
        self._pieces = pieces
        self._transforms = transforms
        
    def snap_together(self):
        """Adjust pieces and transforms so that they fit together.
        
        Fitting together means that if two vertices from different pieces should
        be glued together by this arrangement, then the transformed vertices are
        at the same location.
        """
        # Get center of mass of all pieces
        
        # Loop over all identified vertices
        for identified_vertices in self.find_identified_vertices():
            # Get the transformed vertices
            transformed_vertices = [
                self._transforms[piece].apply(piece.vertices[vertex])
                for piece, vertex in identified_vertices
            ]
            # Find the average of the transformed vertices
            average_vertex = np.mean(transformed_vertices, axis=0)
            
            # Mutate the piece
            
        
        # Loop over pieces, restoring their original center of mass
        raise NotImplementedError
        
    def find_identified_vertices(self):
        """Find vertices that are identified by the arrangement.
        
        If two vertices from different pieces should be glued together by this
        arrangement, then we call them "identified". This is the terminology
        used in topology/geometry.
        
        Returns:
            list[list[(piece, vertex index)]]: List of lists of tuples. Each
                inner list contains tuples of the form (piece, vertex index)
                where the vertices are identified.
        """
        raise NotImplementedError
