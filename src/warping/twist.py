"""Twist warping class."""

import numpy as np
from warping import base_warping


class Twist(base_warping.BaseWarping):
    """Twist warping."""
    
    def __init__(self,
                 puzzle_bounds,
                 num_twists,
                 rotation_magnitude=0.,
                 basin_size=1):
        """Constructor.
        
        Args:
            puzzle_bounds (np.array): Array of shape (2, 2) containing the 
                bounds of the puzzle.
            num_twists (int): Number of twists.
            rotation_magnitude (float): Magnitude of the rotation.
            basin_size (float): Size of the basin of attraction.
        """
        # Sample twist centers
        self._num_twists = num_twists
        self._twist_centers = np.random.uniform(
            low=puzzle_bounds[0],
            high=puzzle_bounds[1],
            size=(num_twists, 2),
        )
        self._rotation_magnitude = rotation_magnitude
        self._basin_size = basin_size
        
        # Sample twist angles
        self._twist_angles = np.random.uniform(
            -self._rotation_magnitude, self._rotation_magnitude,
            size=self._num_twists)
        
    def __call__(self, points: np.array) -> np.array:
        """Apply the warping to a set of points.
        
        Args:
            points (np.array): Array of shape (n, 2) containing the points.
            
        Returns:
            np.array: Array of shape (n, 2) containing the warped points.
        """
        perturbations = np.zeros_like(points)
        for c, a in zip(self._twist_centers, self._twist_angles):
            relative_diffs = points - c[np.newaxis]
            relative_dists = np.linalg.norm(relative_diffs, axis=1)
            twist_amplitude = np.exp(-relative_dists / self._basin_size)
            angles = a * twist_amplitude
            perturbations += twist_amplitude[:, np.newaxis] * np.stack(
                [1 - np.cos(angles), np.sin(angles)], axis=1)
            
        perturbed_points = points + perturbations
        return perturbed_points
    