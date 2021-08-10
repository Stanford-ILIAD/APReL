import numpy as np

def get_random_normalized_vector(dim: int) -> np.array:
    vec = np.random.randn(dim)
    return vec / np.linalg.norm(vec)
