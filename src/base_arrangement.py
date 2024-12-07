"""Base arrangement class."""

from matplotlib import pyplot as plt
import numpy as np
from pieces import base_piece

from typing import List, Tuple


class Transform:
    """Affine transform."""
    
    def __init__(self, translation: np.array, theta: float):
        """Constructor.
        
        Args:
            translation (np.array): Translation vector with 2 elements.
            theta (float): Rotation angle in radians.
        """
        self.translation = translation
        self.rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ])
        self.inverse_rotation_matrix = np.array([
            [np.cos(-theta), -np.sin(-theta)],
            [np.sin(-theta), np.cos(-theta)],
        ])
    
    def apply(self, points: np.array) -> np.array:
        """Apply the transform to a set of points.
        
        Args:
            points (np.array): Array of shape (n, 2) containing the points.
            
        Returns:
            np.array: Array of shape (n, 2) containing the transformed points.
        """
        rotated_points = np.dot(points, self.rotation_matrix)
        translated_points = rotated_points + self.translation
        return translated_points
    
    def inverse(self, points: np.array) -> np.array:
        """Apply the inverse transform to a set of points.
        
        Args:
            points (np.array): Array of shape (n, 2) containing the points.
            
        Returns:
            np.array: Array of shape (n, 2) containing the transformed points.
        """
        translated_points = points - self.translation
        rotated_points = np.dot(
            translated_points, self.inverse_rotation_matrix)
        return rotated_points
    
    
class IdentifiedVertices:
    """Holder class for identified vertices."""
    
    def __init__(self,
                 *piece_vertex_pairs: List[Tuple[base_piece.BasePiece, int]]):
        """Constructor.
        
        Args:
            piece_vertex_pairs: List of (piece, vertex) pair, where vertex is an
                integer index. Each of these piece-vertices will be identified.
        """
        self._piece_vertex_pairs = piece_vertex_pairs
    
    @property
    def pieces(self):
        return [x[0] for x in self._piece_vertex_pairs]
    
    @property
    def vertices(self):
        return [x[1] for x in self._piece_vertex_pairs]
    
    @property
    def piece_vertex_pairs(self):
        return self._piece_vertex_pairs


class BaseArrangement:
    
    def __init__(self,
                 pieces: List[base_piece.BasePiece],
                 transforms: List[Transform]):
        """Constructor.
        
        Args:
            pieces (list[base_piece.BasePiece]): List of pieces.
            transforms (list[Transform]): List of transforms of same length as
                pieces. Each transform tells us how to change coordinates from
                the piece's coordinate frame to the arrangement's coordinate
                frame.
        """
        self._pieces = pieces
        self._transforms = transforms
        self._identified_vertices = self._find_identified_vertices()
        self._num_pieces = len(pieces)
        
    def snap_together(self):
        """Adjust pieces and transforms so that they fit together.
        
        Fitting together means that if two vertices from different pieces should
        be glued together by this arrangement, then the transformed vertices are
        at the same location.
        """
        # Loop over all identified vertices
        for identified_vertices in self._identified_vertices:
            # Get the transformed vertices
            transformed_vertices = [
                self._transforms[piece].apply(piece.vertices[vertex])
                for piece, vertex in identified_vertices.piece_vertex_pairs
            ]
            # Find the average of the transformed vertices
            average_vertex = np.mean(transformed_vertices, axis=0)
            
            # Mutate the pieces in place
            for piece, vertex in identified_vertices.piece_vertex_pairs:
                piece.vertices[vertex] = average_vertex
        
        # Loop over pieces, restoring their original centroid
        for piece, transform in zip(self._pieces, self._transforms):
            # TODO: Change to center of mass instead of average of vertices
            centroid = np.mean(piece.vertices, axis=0)
            piece.vertices -= centroid[None]
            transform.translation += centroid
            
        # TODO: Consider correcting for rotation of the mutated piece
        
    def arrange(self) -> List[np.ndarray]:
        """Apply transforms to all pieces to arrange."""
        arranged_pieces = [
            t.apply(p.vertices) for t, p in zip(self._transforms, self._pieces)
        ]
        return arranged_pieces
        
    def _find_identified_vertices(
        self,
        epsilon=1e-3,
    ) -> List[IdentifiedVertices]:
        """Find vertices that are identified by the arrangement.
        
        If two vertices from different pieces should be glued together by this
        arrangement, then we call them "identified". This is the terminology
        used in topology/geometry.
        
        Returns:
            list[list[(piece, vertex index)]]: List of lists of tuples. Each
                inner list contains tuples of the form (piece, vertex index)
                where the vertices are identified.
        """
        vertex_coordinates = []
        identified_vertices = []
        for piece in self._pieces:
            for vertex_index, vertex in enumerate(piece.vertices):
                # Seed if necessary
                if len(identified_vertices) == 0:
                    vertex_coordinates.append(vertex)
                    identified_vertices.append([(piece, vertex_index)])
                
                # Check if vertex is near vertex_coordinates
                distances = np.array([
                    np.linalg.norm(vertex - x) for x in vertex_coordinates
                ])
                nearby = distances < epsilon
                
                if np.sum(nearby) == 0:
                    # Vertex is not near an already identified vertex, so start
                    # a new identified vertex
                    vertex_coordinates.append(vertex)
                    identified_vertices.append([(piece, vertex_index)])
                elif np.sum(nearby) == 1:
                    # Vertex is near an already identified vertex, so add it to
                    # that one
                    nearby_index = np.argmax(nearby)
                    identified_vertices[nearby_index].append(
                        (piece, vertex_index))
                else:
                    # Vertex is not nearby to more than one identified vertex
                    raise ValueError(
                        f'There exists a piece vertex that is near '
                        f'{np.sum(nearby)} idenficied vertices, but each piece '
                        'vertex must be associated with at most one identified '
                        'vertex.'
                    )
                    
        # Convert identified vertices into IdentifiedVertices datatypes
        identified_vertices = [
            IdentifiedVertices(x) for x in identified_vertices
        ]
        
        return identified_vertices
    
    def plot(self):
        """Plot arranged pieces."""
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Iterate through piece, plotting vertices and edges
        arranged_pieces = self.arrange()
        
        for piece_index in range(self._num_pieces):
            vertices = arranged_pieces[piece_index]
            label = self._pieces[piece_index].label
            
            # Plot edges
            for i in range(len(vertices)):
                ax.plot(
                    [vertices[i, 0], vertices[(i + 1) % len(vertices), 0]],
                    [vertices[i, 1], vertices[(i + 1) % len(vertices), 1]],
                    c='k',
                )
            
            # Plot vertices
            ax.scatter(vertices[:, 0], vertices[:, 1], c='k')
            
            # Plot label
            centroid = np.mean(vertices, axis=0)
            ax.text(centroid[0], centroid[1], label, fontsize=12)
        
        return fig
