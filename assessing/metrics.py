"""Functions that are useful for assessing the accuracy of the given learning agent."""
import numpy as np

from learning import Belief, User


def cosine_similarity(belief: Belief, true_user: User) -> float:
    """
    For testing how well we have learned omega so far, we use a User model based around a true omega,
    and compare our predicted omega to the true omega.
    Args:
        belief: the learning agent's belief distribution over omega
        true_user: a User which has a given true omega

    Returns: the cosine similarity of the predicted omega and the true omega

    """
    omegahat = belief.mean['omega']
    omega = true_user.params['omega']
    return np.dot(omega, omegahat) / (np.linalg.norm(omega) * np.linalg.norm(omegahat))
