"""Interpolation functions."""

from matplotlib import pyplot as plt
import numpy as np


def proximity_interpolation(
    x: np.array,
    y: np.array,
    epsilon: float = 0.1
) -> np.array:
    """Proximity interpolation function from x to y.
    
    Args:
        x (np.array): Array of x values of shape (n_x, 2).
        y (np.array): Array of y values of shape (n_y, 2).
        epsilon (float): Proximity bound.
    
    Returns:
        mapping (np.array): Array of shape (n_y, n_x) where mapping[i, j] is the
            weight of x[j] in the interpolation of y[i].
    """
    n_x = len(x)
    n_y = len(y)
    mapping = np.zeros((n_y, n_x))
    for i in range(n_y):
        y_i = y[i]
        # Compute distance between y[i] and all x
        distances = np.linalg.norm(x - y_i, axis=1)
        
        # Compute weights
        weights = np.maximum(epsilon - distances, 0)
        if np.sum(weights) == 0:
            continue
        weights /= np.sum(weights)
        mapping[i] = weights
    
    return mapping


def nearest_neighbor_interpolation(
    x: np.array,
    y: np.array,
    num_neighbors: int = 10,
    max_epsilon: float = 0.2,
) -> np.array:
    """Nearest neighbor interpolation function from x to y.
    
    Args:
        x (np.array): Array of x values of shape (n_x, 2).
        y (np.array): Array of y values of shape (n_y, 2).
        num_neighbors (int): Number of neighbors to consider.
        max_epsilon (float): Maximum distance to consider.
    
    Returns:
        mapping (np.array): Array of shape (n_y, n_x) where mapping[i, j] is the
            weight of x[j] in the interpolation of y[i].
    """
    n_x = len(x)
    n_y = len(y)
    mapping = np.zeros((n_y, n_x))
    for i in range(n_y):
        y_i = y[i]
        # Compute distance between y[i] and all x
        distances = np.linalg.norm(x - y_i, axis=1)
        
        # Compute weights
        neighbors = np.argsort(distances)[:num_neighbors]
        weights = np.zeros(n_x)
        weights[neighbors] = 1
        weights[distances > max_epsilon] = 0
        if np.sum(weights) == 0:
            continue
        weights /= np.sum(weights)
        mapping[i] = weights
    
    return mapping


def demo():
    """Demo of linear interpolation."""
    # Make random x and y points
    x = np.random.uniform(0, 1, (100, 2))
    y = np.random.uniform(0, 1, (100, 2))
    mapping = proximity_interpolation(x, y, epsilon=0.2)
    
    # Scatterplot of x and y and lines between interpolated points with hue
    # proportional to weight
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(x[:, 0], x[:, 1], color="blue", label="x")
    ax.scatter(y[:, 0], y[:, 1], color="red", label="y")
    for i in range(len(y)):
        for j in range(len(x)):
            ax.plot(
                [x[j, 0], y[i, 0]], [x[j, 1], y[i, 1]],
                alpha=mapping[i, j],
                color='k',
                zorder=0,
            )
    plt.show()
    

if __name__ == "__main__":
    demo()