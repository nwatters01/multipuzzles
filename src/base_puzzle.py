"""Base puzzle class."""

from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from pieces import base_piece
import base_arrangement
import shutil

# Maximum allowable number of iterations for snapping together
_MAX_ITERS = 1000


class BasePuzzle:
    
    def __init__(self, pieces):
        self._pieces = pieces
        self._arrangements = []
        
    def add_arrangement(self, arrangement):
        # Ensure that the arrangement has the same pieces as existing
        # arrangements
        if len(self._arrangements) > 1:
            old_pieces = self._arrangements[0].pieces
            new_pieces = arrangement.pieces
            if len(old_pieces) != len(new_pieces):
                raise ValueError(
                    "Arrangement has different number of pieces than existing "
                    "arrangements."
                )
            for old_piece, new_piece in zip(old_pieces, new_pieces):
                if id(old_piece) != id(new_piece):
                    raise ValueError(
                        "Arrangement has different pieces than existing "
                        "arrangements."
                    )
        
        # Add arrangement
        self._arrangements.append(arrangement)
        
    def add_warping(self,
                    warping,
                    arrangement_index,
                    convergence_threshold=1e-2,
                    plot_every=None):
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
                transform_0.apply(wibbled_edge))[::-1]
            piece_1 = self._pieces[piece_index_1]
            piece_1.edges[edge_index_1] = wibbled_edge_1
            already_wibbled[piece_index_1][edge_index_1] = True
            self._propagate_wibbled_edge(
                piece_index_1, edge_index_1, wibbled_edge_1, already_wibbled)
    
    def wibble_edges(self, wibbled_path_object):
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
                
    def get_arrangement_pixel_mapping(
        self,
        arrangement_index_0,
        image_size_0,
        arrangement_index_1,
        image_size_1,
        num_neighbors=5,
    ):
        arrangement_0 = self._arrangements[arrangement_index_0]
        arrangement_1 = self._arrangements[arrangement_index_1]
        arrangement_0_to_pieces = arrangement_0.image_pixels_to_piece_pixels(
            image_size_0, num_neighbors=num_neighbors)
        pieces_to_arrangement_1 = arrangement_1.piece_pixels_to_image_pixels(
            image_size_1, num_neighbors=num_neighbors)
        composition = pieces_to_arrangement_1 @ arrangement_0_to_pieces
        # import pdb; pdb.set_trace()
        return composition
        
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
    
    def save(self, log_dir, zfill=4):
        shutil.rmtree(log_dir, ignore_errors=True)
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving to {log_dir}")
        
        # Save pieces
        log_dir_pieces = log_dir / "pieces"
        for i, piece in enumerate(self._pieces):
            piece.save(log_dir_pieces / str(i).zfill(zfill))
        
        # Save arrangements
        log_dir_arrangements = log_dir / "arrangements"
        for i, arrangement in enumerate(self._arrangements):
            arrangement.save(log_dir_arrangements / str(i).zfill(zfill))
            
    @classmethod
    def load(cls, log_dir):
        """Recreate from log directory."""
        log_dir = Path(log_dir)
        
        # Load pieces
        log_dir_pieces = log_dir / "pieces"
        pieces = []
        for piece_file in sorted(list(log_dir_pieces.iterdir())):
            piece = base_piece.BasePiece.load(piece_file)
            pieces.append(piece)
            
        # Load the arrangements
        log_dir_arrangements = log_dir / "arrangements"
        arrangements = []
        for arrangement_dir in sorted(list(log_dir_arrangements.iterdir())): 
            arrangement = base_arrangement.BaseArrangement.load(
                pieces, arrangement_dir)
            arrangements.append(arrangement)
        
        puzzle = cls(pieces)
        for arrangement in arrangements:
            puzzle.add_arrangement(arrangement)
        
        return puzzle