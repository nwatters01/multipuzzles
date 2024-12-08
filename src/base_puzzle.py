"""Base puzzle class."""

from matplotlib import pyplot as plt
import numpy as np

# Maximum allowable number of iterations for snapping together
_MAX_ITERS = 1000
# _MAX_ITERS = 1


class BasePuzzle:
    
    def __init__(self, pieces):
        self._pieces = pieces
        self._arrangements = []
        
    def add_arrangement(self, arrangement):
        self._arrangements.append(arrangement)
        
    def add_warping(self,
                    warping,
                    arrangement_index,
                    convergence_threshold=1e-2,
                    plot_every=None):
        # TODO: Consider moving this to the arrangement class
        
        # Apply the warping to the arrangement
        arrangement = self._arrangements[arrangement_index]
        for piece, transform in zip(arrangement.pieces, arrangement.transforms):
            transformed_vertices = transform.apply(piece.vertices)
            warped_vertices = warping(transformed_vertices)
            new_vertices = transform.inverse(warped_vertices)
            new_centroid = np.mean(new_vertices, axis=0)
            piece.vertices = new_vertices - new_centroid
            transform.translation += (
                transform.inverse_rotation_matrix @ new_centroid)
        
        # Reorder arrangements to start at arrangement_index. This will be the 
        # order of snapping together for each iteration.
        arrangements = (
            self._arrangements[arrangement_index:] +
            self._arrangements[:arrangement_index]
        )
        # Snap together for all arrangements
        done = False
        for iteration in range(_MAX_ITERS):
            worst_error = np.max([a.snap_together() for a in arrangements])
            
            # Plot if necessary
            if plot_every is not None and iteration % plot_every == 0:
                self.plot_arrangements(f"Iteration {iteration}; ")
            
            if worst_error < convergence_threshold:
                done = True
                print(f"Converged after {iteration} iterations.")
                break
        
        # Sanity check for convergence
        if not done:
            print(f"Did not converge. Worst error is {worst_error}.")
            self.plot_arrangements(f"Iteration {iteration}; ")
            plt.show()
            raise ValueError(f"Did not converge. Worst error is {worst_error}.")
        
    def plot_arrangements(self, title_prefix=""):
        num_arrangements = len(self._arrangements)
        fig, axes = plt.subplots(
            num_arrangements, 1, figsize=(8, num_arrangements * 4),
            sharex=True, sharey=True,
        )
        if num_arrangements == 1:
            axes = [axes]
        
        for i in range(num_arrangements):
            title = f"{title_prefix}Arrangement {i}"
            self._arrangements[i].plot(axes[i], title)
        
        fig.tight_layout()
        
        return fig