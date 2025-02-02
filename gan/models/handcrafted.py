"""Radial loss demo."""

from models import base_generator
import numpy as np
import torch


class HandcraftedGenerator(base_generator.BaseGenerator):
    
    def __init__(self,
                 square_size,
                 circle_radius,
                 puzzle_log_dir,
                 arrangement_indices,
                 image_sizes):
        """Initialize the generator."""
        super().__init__(
            puzzle_log_dir=puzzle_log_dir,
            arrangement_indices=arrangement_indices,
            image_sizes=image_sizes,
        )
        
        self._loss_fn = torch.nn.MSELoss()
        
        # Make red square image in numpy
        red_square = np.zeros(image_sizes[0] + (3,), dtype=np.float32)
        midpoints = np.array(image_sizes[0]) // 2
        red_square[
            midpoints[0] - square_size // 2:midpoints[0] + square_size // 2,
            midpoints[1] - square_size // 2:midpoints[1] + square_size // 2,
            0,
        ] = 1.
        self._target_base_image = torch.from_numpy(red_square)
        
        # Make blue circle image in numpy
        blue_circle = np.zeros(image_sizes[1] + (3,), dtype=np.float32)
        center = np.array(image_sizes[1]) // 2
        meshgrid = np.meshgrid(
            np.arange(image_sizes[1][0]), np.arange(image_sizes[1][1]),
        )
        distances = np.linalg.norm(
            np.stack(meshgrid) - center[:, None, None], axis=0)
        blue_circle[distances < circle_radius, 2] = 1.
        self._target_mapped_image = torch.from_numpy(blue_circle)
    
    def loss_per_arrangement(self, base_image, mapped_image):
        """Return a loss term for each arrangement."""
        base_image_loss = self._loss_fn(base_image, self._target_base_image)
        mapped_image_loss = self._loss_fn(
            mapped_image, self._target_mapped_image)
        return base_image_loss, mapped_image_loss