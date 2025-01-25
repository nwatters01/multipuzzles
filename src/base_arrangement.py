"""Base arrangement class."""

import interpolation
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
    
    def __init__(self, *piece_vertex_pairs: List[Tuple[int, int]]):
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
    
    
class IdentifiedEdges:
    """Holder class for identified edges."""
    
    def __init__(self, *piece_edge_pairs: List[Tuple[int, int]]):
        """Constructor.
        
        Args:
            piece_edge_pairs: List of (piece, edge) pair, where edge is an
                integer index. Each of these piece-edges will be identified.
        """
        self._piece_edge_pairs = piece_edge_pairs
    
    @property
    def pieces(self):
        return [x[0] for x in self._piece_edge_pairs]
    
    @property
    def edges(self):
        return [x[1] for x in self._piece_edge_pairs]
    
    @property
    def piece_edge_pairs(self):
        return self._piece_edge_pairs


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
        self._identified_edges = self._find_identified_edges()
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
    
    def _find_identified_edges(
        self,
        epsilon=1e-3,
    ) -> List[IdentifiedEdges]:
        """Find edges that are identified by the arrangement.
        
        If two edges from different pieces should be glued together by this
        arrangement, then we call them "identified". This is the terminology
        used in topology/geometry.
        
        Returns:
            list[list[(piece, edge index)]]: List of lists of tuples. Each
                inner list contains tuples of the form (piece, edge index)
                where the edges are identified.
        """
        identified_edges = {}
        
        # Compute endpoint positions of each edges
        edge_to_endpoints = {}
        arranged_piece_vertices = self.arrange()
        for piece_index, piece in enumerate(self._pieces):
            for edge_index in range(piece.num_sides):
                endpoint_vertex_indices = piece.vertex_indices_per_edge[
                    edge_index]
                edge_to_endpoints[(piece_index, edge_index)] = [
                    arranged_piece_vertices[piece_index][i]
                    for i in endpoint_vertex_indices
                ]
        
        # Loop through edges, finding others that share endpoint positions
        for key_0, endpoints_0 in edge_to_endpoints.items():
            for key_1, endpoints_1 in edge_to_endpoints.items():
                if key_0 == key_1:
                    continue
                
                # Check if endpoints are the same
                endpoint_distances = [
                    np.linalg.norm(e0 - e1)
                    for e0, e1 in zip(endpoints_0, endpoints_1[::-1])
                ]
                if np.all(np.array(endpoint_distances) < epsilon):
                    identified_edges[key_0] = key_1
                    identified_edges[key_1] = key_0
                    
        return identified_edges
    
    def plot(self, ax, title=''):
        """Plot arranged pieces."""
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        
        # Iterate through piece, plotting vertices, edges, and pixel positions
        arranged_pieces = self.arrange()
        
        for piece_index in range(self._num_pieces):
            piece = self._pieces[piece_index]
            vertices = arranged_pieces[piece_index]
            label = piece.label
            transform = self._transforms[piece_index]
            
            # Plot edges
            for edge in piece.edges:
                transformed_edge = transform.apply(edge)
                ax.plot(
                    transformed_edge[:, 0],
                    transformed_edge[:, 1],
                    c='k',
                    linewidth=1,
                )
            
            # Plot vertices
            ax.scatter(vertices[:, 0], vertices[:, 1], c='k', s=10)
            
            # Plot pixels
            pixel_positions = transform.apply(piece.pixel_positions)
            ax.scatter(
                pixel_positions[:, 0],
                pixel_positions[:, 1],
                c=piece.pixel_values,
                s=0.3,
            )
            
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
        
    def get_image_pixels(self, image_size: Tuple[int, int]):
        """Get pixels of image spanning the arrangement."""
        bounding_box = self.bounding_box
        x = np.linspace(bounding_box[0, 0], bounding_box[1, 0], image_size[0])
        y = np.linspace(bounding_box[0, 1], bounding_box[1, 1], image_size[1])
        image_pixel_positions = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
        return image_pixel_positions
    
    def get_piece_pixels(self):
        """Get pixels of all the pieces."""
        piece_pixels = []
        for piece_index in range(self._num_pieces):
            piece = self._pieces[piece_index]
            transform = self._transforms[piece_index]
            transformed_pixel_positions = transform.apply(piece.pixel_positions)
            piece_pixels.append(transformed_pixel_positions)
        piece_pixels = np.concatenate(piece_pixels)
        return piece_pixels
    
    def piece_pixels_to_image_pixels(
        self,
        image_size: Tuple[int, int],
        num_neighbors: int = 10,
    ) -> np.array:
        """Linear map from piece pixel positions to image pixel positions.
        
        Args:
            image_size (tuple[int, int]): Size of the image.
        
        Returns:
            mapping (np.array): Array of shape (n_piece_pixels, n_image_pixels).
        """
        piece_pixels = self.get_piece_pixels()
        image_pixels = self.get_image_pixels(image_size)
        mapping = interpolation.nearest_neighbor_interpolation(
            piece_pixels, image_pixels, num_neighbors=num_neighbors)
        return mapping

    def image_pixels_to_piece_pixels(
        self,
        image_size: Tuple[int, int],
        num_neighbors: int = 10,
    ) -> np.array:
        """Linear map from image pixel positions to piece pixel positions.
        
        Args:
            image_size (tuple[int, int]): Size of the image.
        
        Returns:
            mapping (np.array): Array of shape (n_image_pixels, n_piece_pixels).
        """
        piece_pixels = self.get_piece_pixels()
        image_pixels = self.get_image_pixels(image_size)
        mapping = interpolation.nearest_neighbor_interpolation(
            image_pixels, piece_pixels, num_neighbors=num_neighbors)
        return mapping
            
    @property
    def pieces(self):
        return self._pieces
    
    @property
    def transforms(self):
        return self._transforms
    
    @property
    def bounding_box(self):
        """Return the bounding box of the arrangement."""
        # Get arranged edges
        arranged_edges = []
        for piece_index in range(self._num_pieces):
            piece = self._pieces[piece_index]
            transform = self._transforms[piece_index]
            arranged_edges.extend([
                transform.apply(edge) for edge in piece.edges
            ])
        arranged_edges = np.concatenate(arranged_edges)
        
        # Find the bounding box
        min_x, min_y = np.min(arranged_edges, axis=0)
        max_x, max_y = np.max(arranged_edges, axis=0)
        return np.array([[min_x, min_y], [max_x, max_y]])
    
    @property
    def identified_vertices(self):
        return self._identified_vertices
    
    @property
    def identified_edges(self):
        return self._identified_edges
