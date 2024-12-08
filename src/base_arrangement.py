"""Base arrangement class."""

import numpy as np
from pieces import base_piece

from typing import List, Tuple


def best_fit_rotation(x, y):
    """Find best fit rotation matrix between two 2-dimensional datasets.
    
    Args:
        x (np.array): Array of shape (n, 2) containing the first set of points.
        y (np.array): Array of shape (n, 2) containing the second set of points.
    """
    # Center the points by subtracting the centroid of each set of points
    x_centered = x - np.mean(x, axis=0)
    y_centered = y - np.mean(y, axis=0)
    
    # Compute SVD of the covariance matrix
    cov = x_centered.T @ y_centered
    U, _, Vh = np.linalg.svd(cov)
    rotation_matrix = Vh.T @ U.T
    
    # Ensure a proper rotation (i.e., no reflection)
    if np.linalg.det(rotation_matrix) < 0:
        Vh[-1, :] *= -1
        rotation_matrix = Vh.T @ U.T
    
    return rotation_matrix


class Transform:
    """Affine transform."""
    
    def __init__(self, translation: np.array, theta: float):
        """Constructor.
        
        Args:
            translation (np.array): Translation vector with 2 elements.
            theta (float): Rotation angle in radians.
        """
        self.translation = translation
        self._theta = theta
        self.make_rotation_matrices()
        
    def make_rotation_matrices(self):
        self.rotation_matrix = np.array([
            [np.cos(self._theta), -np.sin(self._theta)],
            [np.sin(self._theta), np.cos(self._theta)],
        ])
        self.inverse_rotation_matrix = np.array([
            [np.cos(-self._theta), -np.sin(-self._theta)],
            [np.sin(-self._theta), np.cos(-self._theta)],
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
    
    @property
    def theta(self):
        return self._theta
    
    @theta.setter
    def theta(self, value):
        self._theta = value
        self.make_rotation_matrices()
    
    
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
        self._num_pieces = len(pieces)
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
        worst_error = 0
        new_vertices_per_piece = [
            np.copy(piece.vertices) for piece in self._pieces
        ]
        for identified_vertices in self._identified_vertices:
            # Get the transformed vertices
            transformed_vertices = [
                self._transforms[piece].apply(
                    self._pieces[piece].vertices[vertex])
                for piece, vertex in identified_vertices.piece_vertex_pairs
            ]
            # Find the average of the transformed vertices
            average_vertex = np.mean(transformed_vertices, axis=0)
            
            # Compute the error
            error = np.max([
                np.linalg.norm(average_vertex - x)
                for x in transformed_vertices
            ])
            worst_error = max(worst_error, error)
            
            # Mutate the pieces in place
            for piece, vertex in identified_vertices.piece_vertex_pairs:
                new_vertices_per_piece[piece][vertex] = (
                    self._transforms[piece].inverse(average_vertex))
                
        # Loop over pieces and transforms, finding the best possible new
        # transform
        for i in range(self._num_pieces):
            new_vertices = new_vertices_per_piece[i]
            piece = self._pieces[i]
            transform = self._transforms[i]
            
            # Correct for translation of the piece
            new_centroid = np.mean(new_vertices, axis=0)
            piece.vertices -= new_centroid[None]
            delta_translation= transform.inverse_rotation_matrix @ new_centroid
            transform.translation += delta_translation
            
            # Correct for rotation of the piece
            rotation_matrix = best_fit_rotation(piece.vertices, new_vertices)
            theta = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            transform.theta = transform.theta - theta
            
            # Mutate the vertices to be the new vertices
            new_vertices = new_vertices - new_centroid[None]
            new_v = np.dot(new_vertices, rotation_matrix)
            piece.vertices = new_v
            
        return worst_error
        
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
        arranged_pieces = self.arrange()
        for piece_index in range(self._num_pieces):
            arranged_piece = arranged_pieces[piece_index]
            for vertex_index, vertex in enumerate(arranged_piece):
                piece_vertex_pair = (piece_index, vertex_index)
                # Seed if necessary
                if len(identified_vertices) == 0:
                    vertex_coordinates.append(vertex)
                    identified_vertices.append([piece_vertex_pair])
                    continue
                
                # Check if vertex is near vertex_coordinates
                distances = np.array([
                    np.linalg.norm(vertex - x) for x in vertex_coordinates
                ])
                nearby = distances < epsilon
                
                if np.sum(nearby) == 0:
                    # Vertex is not near an already identified vertex, so start
                    # a new identified vertex
                    vertex_coordinates.append(vertex)
                    identified_vertices.append([piece_vertex_pair])
                elif np.sum(nearby) == 1:
                    # Vertex is near an already identified vertex, so add it to
                    # that one
                    nearby_index = np.argmax(nearby)
                    identified_vertices[nearby_index].append(
                        piece_vertex_pair)
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
            IdentifiedVertices(*x) for x in identified_vertices
        ]
        
        return identified_vertices
    
    def plot(self, ax, title=''):
        """Plot arranged pieces."""
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        
        # Iterate through piece, plotting vertices and edges
        arranged_pieces = self.arrange()
        
        for piece_index in range(self._num_pieces):
            vertices = arranged_pieces[piece_index]
            label = self._pieces[piece_index].label
            transform = self._transforms[piece_index]
            
            # Plot edges
            for i in range(len(vertices)):
                ax.plot(
                    [vertices[i, 0], vertices[(i + 1) % len(vertices), 0]],
                    [vertices[i, 1], vertices[(i + 1) % len(vertices), 1]],
                    c='k', linewidth=1,
                )
            
            # Plot vertices
            ax.scatter(vertices[:, 0], vertices[:, 1], c='k', s=10)
            
            # Plot label
            centroid = np.mean(vertices, axis=0)
            ax.text(
                centroid[0],
                centroid[1],
                label,
                fontsize=10,
                rotation=transform.theta * 180 / np.pi,
                horizontalalignment='center',
                verticalalignment='center',
            )
    
    @property
    def pieces(self):
        return self._pieces
    
    @property
    def transforms(self):
        return self._transforms
