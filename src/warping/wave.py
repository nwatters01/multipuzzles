"""Wave warping class."""

import abc
import numpy as np
from warping import base_warping


class Wave(base_warping.BaseWarping):
    """Wave warping."""
    
    def __init__(self, theta, amplitude, frequency):
        """Constructor.
        
        Args:
            theta (float): Angle of the stretch.
            amplitude (float): Amplitude of the wave.
            frequency (float): Frequency of the wave.
        """
        self._theta = theta
        self._amplitude = amplitude
        self._frequency = frequency
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
        
        # Wave along x axis
        points[:, 1] += self._amplitude * np.sin(self._frequency * points[:, 0])
        
        # Rotate back
        points = np.dot(points, self._rotation_matrix)
        
        return points
    