"""Stretch warping class."""

import abc
import numpy as np
from warping import base_warping


class Stretch(base_warping.BaseWarping):
    """Stretch warping."""
    
    def __init__(self, theta, stretch_factor):
        """Constructor.
        
        Args:
            theta (float): Angle of the stretch.
            stretch_factor (float): Stretch factor.
        """
        self._theta = theta
        self._stretch_factor = stretch_factor
        self._rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ])
        
    def __call__(self, points: np.array) -> np.array:
        """Apply the warping to a set of points.
        
        Args:
            points (np.array): Array of shape (n, 2) containing the points.
            
        Returns:
            np.array: Array of shape (n, 2) containing the warped points.
        """
        # Rotate
        points = np.dot(points, self._rotation_matrix.T)
        
        # Stretch along x axis
        points[:, 0] *= self._stretch_factor
        
        # Rotate back
        points = np.dot(points, self._rotation_matrix)
        
        return points
    