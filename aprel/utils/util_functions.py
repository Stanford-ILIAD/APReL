"""General utility functions."""
import numpy as np

def get_random_normalized_vector(dim: int) -> np.array:
    """
    Returns a random normalized vector with the given dimensions.
    
    Args:
        dim (int): The dimensionality of the output vector.
        
    Returns:
        numpy.array: A random normalized vector that lies on the surface of the :py:attr:`dim`-dimensional hypersphere.
    """
    vec = np.random.randn(dim)
    return vec / np.linalg.norm(vec)
