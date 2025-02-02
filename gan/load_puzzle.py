"""Decomposing puzzle."""

from matplotlib import pyplot as plt
import numpy as np

import sys
sys.path.append("../src")
import base_puzzle
from PIL import Image


def load_puzzle_from_snapshot(
    log_dir: str,
    arrangement_indices: list,
    image_sizes: list,
):
    """Load puzzle from snapshot."""
    # Load puzzle
    puzzle = base_puzzle.BasePuzzle.load(log_dir)
    
    # Get pixel mapping from one arrangement to the other
    mapping = puzzle.get_arrangement_pixel_mapping(
        arrangement_indices[0],
        image_sizes[0],
        arrangement_indices[1],
        image_sizes[1],
    )
    
    return puzzle, mapping

        
def main():
    """Run and plot an example."""
    snapshot_dir = "../logs/square_5x5_v0"
    image_path = "../images/frog.jpg"
    image_sizes = [(110, 100), (100, 110)]
    arrangement_indices = [0, 1]
    
    _, mapping = load_puzzle_from_snapshot(
        snapshot_dir, arrangement_indices, image_sizes,
    )

    # Load image as array and transform to the right orientation
    image = plt.imread(image_path)
    image_pil = Image.fromarray(image.astype(np.uint8))
    image = np.array(image_pil.resize(image_sizes[0])) / 255.0
    image = np.transpose(image, (1, 0, 2))
    
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
    main()
        