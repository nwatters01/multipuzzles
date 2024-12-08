"""Base puzzle class."""

from matplotlib import pyplot as plt
import numpy as np

# Maximum allowable number of iterations for snapping together
_MAX_ITERS = 1000
# _MAX_ITERS = 1


def _intersects(path_0, path_1, intersection_thresh=0.04):
    """Detect whether two paths intersect.
    
    This intersection detection is the slowest part of the code. Consider
    alternative algorithms to improve runtime.
    """
    endpoint_buffer_0 = int(np.floor(0.1 * len(path_0)))
    endpoint_buffer_1 = int(np.floor(0.1 * len(path_1)))
    path_0 = path_0[endpoint_buffer_0: -endpoint_buffer_0]
    path_1 = path_1[endpoint_buffer_1: -endpoint_buffer_1]
    path_dists = np.linalg.norm(
        path_0[np.newaxis] - path_1[:, np.newaxis], axis=2)
    if np.sum(path_dists < intersection_thresh):
        return True
    else:
        return False


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
    
    def _propagate_wibbled_edge(self,
                                piece_index,
                                edge_index,
                                wibbled_edge,
                                already_wibbled):
        """Propagate the wibbled edge to all identified edges.
        
        This function is called recursively to propagate the wibbled edge to all
        identified edges under all arrangements. This is needed because one
        wibbled edge affects pieces of other arrangements, which in turn affect
        more pieces in the original arrangement, et cetera.
        """
        for arrangement in self._arrangements:
            identified_edges = arrangement.identified_edges
            if not (piece_index, edge_index) in identified_edges:
                continue
            
            piece_index_1, edge_index_1 = identified_edges[
                (piece_index, edge_index)
            ]
            if already_wibbled[piece_index_1][edge_index_1]:
                continue
            transform_0 = arrangement.transforms[piece_index]
            transform_1 = arrangement.transforms[piece_index_1]
            wibbled_edge_1 = transform_1.inverse(
                transform_0.apply(wibbled_edge))
            piece_1 = self._pieces[piece_index_1]
            piece_1.edges[edge_index_1] = wibbled_edge_1
            already_wibbled[piece_index_1][edge_index_1] = True
            self._propagate_wibbled_edge(
                piece_index_1, edge_index_1, wibbled_edge_1, already_wibbled)
    
    def wibble_edges(self, wibbled_path_object, intersection_thresh=0.04):
        """Wibble the edges of the pieces."""
        
        # Iterate through edges of each piece
        already_wibbled = [
            [False for _ in piece.edges] for piece in self._pieces
        ]
        for piece_index_0, piece_0 in enumerate(self._pieces):
            for edge_index_0, edge_0 in enumerate(piece_0.edges):
                # Skip if already wibbled
                if already_wibbled[piece_index_0][edge_index_0]:
                    continue
                
                # Generate wibbled path
                wibbled_edge = wibbled_path_object(
                    start=edge_0[0], end=edge_0[1])
                piece_0.edges[edge_index_0] = wibbled_edge
                already_wibbled[piece_index_0][edge_index_0] = True
                self._propagate_wibbled_edge(
                    piece_index_0, edge_index_0, wibbled_edge, already_wibbled)
        
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