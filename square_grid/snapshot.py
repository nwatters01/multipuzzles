"""Decomposing puzzle."""

from matplotlib import pyplot as plt
import numpy as np

import sys
sys.path.append("../src")
import base_puzzle
import base_arrangement
from warping import stretch
from warping import twist
from warping import wave
from pieces import square_piece
import edge_wibble
from PIL import Image

_RANDOM_SEED = 4


class SnapshotPuzzle(base_puzzle.BasePuzzle):
        
    def interpolation(self, arrangement_indices, image_sizes, image_path):
        """Render image interpolation."""
        
        # Load image as array and transform to the right orientation
        image = plt.imread(image_path)
        
        # Resize image to the image_sizes[0]
        image_pil = Image.fromarray(image.astype(np.uint8))
        image = np.array(image_pil.resize(image_sizes[0])) / 255.0
        # Transpose image because PIL reverses the order of the axes
        image = np.transpose(image, (1, 0, 2))
        
        # Get pixel mapping from one arrangement to the other
        mapping = self.get_arrangement_pixel_mapping(
            arrangement_indices[0],
            image_sizes[0],
            arrangement_indices[1],
            image_sizes[1],
        )
        
        # Flatten image and apply mapping
        image_flat = image.reshape(-1, 3)
        image_flat_interpolated = mapping @ image_flat
        
        # Reshape to image_sizes[1]
        image_interpolated = image_flat_interpolated.reshape(
            image_sizes[1][0], image_sizes[1][1], 3)
        
        # Plot images
        _, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[1].imshow(image_interpolated)
        axes[1].set_title("Interpolated Image")
        plt.show()
        

if __name__ == "__main__":
    np.random.seed(_RANDOM_SEED)
    puzzle = SnapshotPuzzle.load("../logs/decomposing_puzzle_v0")
    # puzzle.interpolation(
    #     [0, 1], [(110, 100), (100, 110)], "../images/frog.jpg",
    # )
    puzzle.plot_arrangements()
    plt.show()
        