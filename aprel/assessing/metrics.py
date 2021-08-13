"""Functions that are useful for assessing the accuracy of the given learning agent."""
import numpy as np

from aprel.learning import LinearRewardBelief, User


def cosine_similarity(belief: LinearRewardBelief, true_user: User) -> float:
    """
    This function tests how well the belief models the true user, when the reward model is linear.
    It performs this test by returning the cosine similarity between the true and predicted reward weights (omega's).
    
    Args:
        belief (LinearRewardBelief): the learning agent's belief about the user
        true_user (User): a User which has a given true omega

    Returns:
        float: the cosine similarity of the predicted omega and the true omega

    """
    omegahat = belief.mean['omega']
    omega = true_user.params['omega']
    return np.dot(omega, omegahat) / (np.linalg.norm(omega) * np.linalg.norm(omegahat))
