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


class DecomposingPuzzle(base_puzzle.BasePuzzle):

    def __init__(self,
                 big_size=(9, 6),
                 decomposed_sizes=((3, 6), (3, 6), (3, 6))):
        """Constructor."""
        self._big_size = big_size
        self._decomposed_sizes = decomposed_sizes
        self._num_pieces = big_size[0] * big_size[1]
        
        # Sanity check that the sizes are consistent
        sum_decomposed_sizes = sum([
            decomposed_size[0] * decomposed_size[1]
            for decomposed_size in decomposed_sizes
        ])
        if sum_decomposed_sizes != self._num_pieces:
            raise ValueError(
                f"Decomposed sizes {sum_decomposed_sizes} do not match big "
                f"size {self._num_pieces}."
            )
        
        # Create all the pieces
        pieces = [
            square_piece.SquarePiece(side_length=1, resolution=10, label=str(i))
            for i in range(self._num_pieces)
        ]
        
        # Create the puzzle
        super().__init__(pieces)
        
        # Create the big arrangement
        self.add_big_arrangement()
        
        # Create the decomposed arrangements
        self.add_decomposed_arrangement()
        
        # Add warpings
        twist_warping = twist.Twist(
            puzzle_bounds=np.array([[0, 0], list(self._big_size)]),
            num_twists=15,
            rotation_magnitude=0.4 * np.pi,
            basin_size=2,
        )
        self.add_warping(twist_warping, arrangement_index=0)
        wave_warping = wave.Wave(theta=0.1 * np.pi, amplitude=0.8, frequency=1)
        self.add_warping(wave_warping, arrangement_index=0)
        stretch_warping = stretch.Stretch(theta=np.pi / 4, stretch_factor=1.3)
        self.add_warping(stretch_warping, arrangement_index=0)
        
        # Add edge wibbling
        sample_noise=edge_wibble.get_sample_noise()
        curvature_object = edge_wibble.Curvature(
            sample_curvature_magnitude=(
                edge_wibble.get_sample_curvature_magnitude()),
            sample_curve_length=edge_wibble.get_sample_curve_length(),
            sample_straight_length=edge_wibble.get_sample_straight_length(),
            sample_noise=sample_noise,
            resolution=500,
        )
        flattening_fn = edge_wibble.get_flattening_fn()
        wibbled_path_object = edge_wibble.WibbledPath(
            curvature_object, flattening_fn)
        self.wibble_edges(wibbled_path_object)
        
        # Update pixel positions and values
        for piece in self._pieces:
            piece.pixelate()
        
        # Render the image
        # self._render_image(0, "../images/frog.jpg")
        
        # Demonstrate the interpolation on the first two arrangements
        self._interpolation(
            [0, 1], [(150, 100), (100, 150)], "../images/frog.jpg",
        )
        
    def _interpolation(self, arrangement_indices, image_sizes, image_path):
        """Render image interpolation."""
        
        # Load image as array and transform to the right orientation
        image = plt.imread(image_path)
        
        # Resize image to the image_sizes[0]
        image_pil = Image.fromarray(image.astype(np.uint8))
        image = np.array(image_pil.resize(image_sizes[0])) / 255.0
        
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
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[1].imshow(image_interpolated)
        axes[1].set_title("Interpolated Image")
        plt.show()
            
    def _render_image(self, arrangement_index, image_path):
        """Render image under given arrangement."""
        arrangement = self._arrangements[arrangement_index]
        
        # Load image as array and transform to the right orientation
        image = plt.imread(image_path)
        image = np.transpose(image, (1, 0, 2))
        image = np.fliplr(image)
        image_size = np.array(image.shape[:2])
        
        # Figure out bounding box of the arrangement
        bounding_box = arrangement.bounding_box
        
        # Iterate through pieces, figuring out where they go and overwriting
        # their pixel values
        for piece_index, piece in enumerate(arrangement.pieces):
            transform = arrangement.transforms[piece_index]
            transformed_pixel_positions = transform.apply(piece.pixel_positions)
            
            # Convert to image coordinates using the bounding box and image size
            transformed_pixel_positions -= bounding_box[0]
            transformed_pixel_positions /= (bounding_box[1] - bounding_box[0])
            image_pixel_positions = transformed_pixel_positions * image_size
            
            # Round and clip pixel positions
            image_pixel_positions = np.clip(
                np.round(image_pixel_positions).astype(int),
                0,
                image_size - 1,
            )
            
            # Get pixel values
            pixel_values = image[
                image_pixel_positions[:, 0],
                image_pixel_positions[:, 1],
            ]
            pixel_values = pixel_values / 255.0
            
            # Set pixel values in the piece
            piece.pixel_values = pixel_values
        
    def add_big_arrangement(self):
        """Add a random arrangement."""
        
        # Find target location for each piece
        target_positions = [
            (i, j)
            for i in range(self._big_size[0])
            for j in range(self._big_size[1])
        ]
        np.random.shuffle(target_positions)
        
        # Create transforms
        transforms = []
        for piece_index in range(self._num_pieces):
            theta = np.random.choice([i * np.pi / 2 for i in range(4)])
            translation = np.array(target_positions[piece_index], dtype=float)
            transform = base_arrangement.Transform(
                translation=translation,
                theta=theta,
            )
            transforms.append(transform)
            
        # Create arrangement
        arrangement = base_arrangement.BaseArrangement(
            pieces=self._pieces, transforms=transforms
        )
        self.add_arrangement(arrangement)
        
    def add_decomposed_arrangement(self, buffer_size=2):
        """Add decomposed arrangements."""
        
        # Target positions
        target_positions = []
        x_offset = 0
        for decomposed_size in self._decomposed_sizes:
            target_positions.extend([
                (x_offset + i, j)
                for i in range(decomposed_size[0])
                for j in range(decomposed_size[1])
            ])
            x_offset += decomposed_size[0] + buffer_size
            
        # Shuffle target positions
        np.random.shuffle(target_positions)
            
        # Get transform per target position
        transforms = []
        for piece_index in range(self._num_pieces):
            theta = np.random.choice([i * np.pi / 2 for i in range(4)])
            translation = np.array(target_positions[piece_index], dtype=float)
            transform = base_arrangement.Transform(
                translation=translation,
                theta=theta,
            )
            transforms.append(transform)
        
        # Create arrangement
        arrangement = base_arrangement.BaseArrangement(
            pieces=self._pieces, transforms=transforms
        )
        self.add_arrangement(arrangement)
        

if __name__ == "__main__":
    np.random.seed(_RANDOM_SEED)
    puzzle = DecomposingPuzzle()
    puzzle.plot_arrangements()
    plt.show()
        