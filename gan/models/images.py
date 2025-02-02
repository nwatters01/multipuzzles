"""Radial loss demo."""

from matplotlib import pyplot as plt
from models import base_generator
import numpy as np
from PIL import Image
import torch


class ImagesGenerator(base_generator.BaseGenerator):
    
    def __init__(self,
                 image_0_path,
                 image_1_path,
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
        
        images = []
        for image, image_size in zip([image_0_path, image_1_path], image_sizes):
            image = plt.imread(image)
            # Resize image to the image_sizes[0]
            image_pil = Image.fromarray(image.astype(np.uint8))
            image = np.array(image_pil.resize(image_size)) / 255.0
            # Transpose image because PIL reverses the order of the axes
            image = np.transpose(image, (1, 0, 2))
            images.append(image.astype(np.float32))
        self._target_images = [torch.from_numpy(image) for image in images]
    
    def loss_per_arrangement(self, base_image, mapped_image):
        """Return a loss term for each arrangement."""
        base_image_loss = self._loss_fn(base_image, self._target_images[0])
        mapped_image_loss = self._loss_fn(mapped_image, self._target_images[1])
        return base_image_loss, mapped_image_loss