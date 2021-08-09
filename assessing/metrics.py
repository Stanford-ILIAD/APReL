import numpy as np

from learning import Belief, User


def cosine_similarity(belief: Belief, true_user: User) -> float:
    omegahat = belief.mean['omega']
    omega = true_user.params['omega']
    return np.dot(omega, omegahat) / (np.linalg.norm(omega) * np.linalg.norm(omegahat))
