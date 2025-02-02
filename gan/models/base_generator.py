"""Base generator."""

import abc
import load_puzzle
import matplotlib.pyplot as plt
import torch


class BaseGenerator(torch.nn.Module, metaclass=abc.ABCMeta):
    
    def __init__(self,
                 puzzle_log_dir,
                 arrangement_indices,
                 image_sizes):
        """Initialize the generator."""
        super().__init__()
        self._image_sizes = image_sizes
        
        # Validate we only have two arrangements
        assert len(arrangement_indices) == 2
        assert len(image_sizes) == 2
        
        # Register number of pixels
        self._pixels_per_image = [x[0] * x[1] for x in image_sizes]
        
        # Load puzzle
        self._puzzle, mapping = load_puzzle.load_puzzle_from_snapshot(
            puzzle_log_dir, arrangement_indices, image_sizes,
        )
        
        # mapping has shape [self._pixels_per_image[1], self._pixels_per_image[0]]
        self._mapping = torch.tensor(mapping, dtype=torch.float32)
        
        # Make initial parameters for the first image
        self._base_image = torch.nn.Parameter(
            torch.rand(*image_sizes[0], 3), requires_grad=True,
        )
        
    @abc.abstractmethod
    def loss_per_arrangement(self, *images):
        """Return a loss term for each arrangement."""
        raise NotImplementedError
    
    def loss(self):
        """Compute loss for the generator."""
        images = self.forward()
        loss_per_arrangement = self.loss_per_arrangement(*images)
        return sum(loss_per_arrangement)
    
    def forward(self):
        """Apply mapping, return one image per arrangement."""
        base_image = self._base_image
        base_image_flat = base_image.view(-1, 3)
        mapped_image_flat = self._mapping @ base_image_flat
        mapped_image = mapped_image_flat.view(*self._image_sizes[1], 3)
        return base_image, mapped_image
    
    def plot(self):
        """Plot the images."""
        base_image, mapped_image = self.forward()
        base_image = base_image.detach().numpy()
        mapped_image = mapped_image.detach().numpy()
        
        _, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(base_image)
        axes[0].set_title("Base Image")
        axes[1].imshow(mapped_image)
        axes[1].set_title("Mapped Image")
        plt.show()