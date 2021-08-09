from typing import Dict
import numpy as np

def uniform_logprior(params: Dict) -> float:
    if np.linalg.norm(params['omega']) <= 1:
        return 0.
    return -np.inf


def gaussian_proposal(point: Dict) -> Dict:
    next_point = {}
    for key, value in point.items():
        if getattr(value, "shape", None) is not None:
            shape = list(value.shape)
        elif isinstance(value, list):
            shape = np.array(value).shape
        else:
            shape = [1]
        next_point[key] = value + np.random.randn(*shape) * 0.05
        if key == 'omega':
            next_point[key] /= np.linalg.norm(next_point[key])
    return next_point
