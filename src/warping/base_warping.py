"""Base warping class."""

import abc
import numpy as np


class BaseWarping(abc.ABC):
    """Base warping class."""
    
    @abc.abstractmethod
    def __call__(self, points: np.array) -> np.array:
        """Apply the warping to a set of points.
        
        Args:
            points (np.array): Array of shape (n, 2) containing the points.
            
        Returns:
            np.array: Array of shape (n, 2) containing the warped points.
        """
        pass
    
    