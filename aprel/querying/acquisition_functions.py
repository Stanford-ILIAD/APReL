"""
This module contains a set of acquisition functions that determine the value of a given query,
which is useful for acitive query optimization.
"""
from typing import List, Dict
import numpy as np

from aprel.basics import Trajectory
from aprel.learning import Belief, SamplingBasedBelief
from aprel.learning import Query, PreferenceQuery, WeakComparisonQuery, FullRankingQuery

def mutual_information(belief: Belief, query: Query, **kwargs) -> float:
    """
    This function returns the mutual information between the given belief distribution and the
    query. Maximum mutual information is often desired for data-efficient learning.
    
    This is implemented based on the following paper:
        - `Asking Easy Questions: A User-Friendly Approach to Active Reward Learning <https://arxiv.org/abs/1910.04365>`_
    
    Args:
        belief (Belief): the current belief distribution over the reward function
        query (Query): a query to ask the user
        **kwargs: none used currently

    Returns:
        float: the mutual information value (always nonnegative)
    """
    if isinstance(belief, SamplingBasedBelief):
        user = belief.user_model.copy()
        probs = []
        for sample in belief.samples:
            user.params = sample
            probs.append(user.response_probabilities(query))
        M = len(probs)
        probs = np.array(probs)
        return np.sum(probs * np.log2(M * probs / np.sum(probs, axis=0))) / M
    
    raise NotImplementedError


def volume_removal(belief: Belief, query: Query, **kwargs) -> float:
    """
    This function returns the expected volume removal from the `unnormalized` belief
    distribution. Maximum volume removal is often desired for data-efficient learning.
    
    This is implemented based on the following two papers:
        - `Active Preference-Based Learning of Reward Functions <http://m.roboticsproceedings.org/rss13/p53.pdf>`_
        - `The Green Choice: Learning and Influencing Human Decisions on Shared Roads <https://arxiv.org/abs/1904.02209>`_
    
    :Note: As `Bıyık et al. (2019) <https://arxiv.org/abs/1910.04365>`_ pointed out, volume
        removal has trivial global maximizers when query maximizes the uncertainty for the
        user, e.g., when all trajectories in the slate of a PreferenceQuery is identical.
        Hence, the optimizations with volume removal are often ill-posed.
    
    Args:
        belief (Belief): the current belief distribution over the reward function
        query (Query): a query to ask the user
        **kwargs: none used currently

    Returns:
        float: the expected volume removal value (always nonnegative)
    """
    if isinstance(belief, SamplingBasedBelief):
        user = belief.user_model.copy()
        probs = []
        for sample in belief.samples:
            user.params = sample
            probs.append(user.response_probabilities(query))
        M = len(probs)
        return np.sum((np.sum(probs, axis=0) / M) ** 2)
        
    raise NotImplementedError


def disagreement(weights: np.array, logprobs: List[float], **kwargs) -> float:
    """
    This function returns the disagreement value between two sets of reward weights (weights's).
    This is useful as an acquisition function when a trajectory planner is available and when
    the desired query contains only two trajectories. The pair of weights with the highest
    disagreement is found and then the best trajectories according to them forms the optimized query.
    
    This is implemented based on the following paper:
        - `Learning an Urban Air Mobility Encounter Model from Expert Preferences <https://arxiv.org/abs/1907.05575>`_
    
    Args:
        weights (numpy.array): 2 x d array where each row is a set of reward weights. The
            disagreement between these two weights will be calculated.
        logprobs (List[float]): log probabilities of the given reward weights under the belief.
        **kwargs: acquisition function hyperparameters:

            - `lambda` (float) The tradeoff parameter. The higher lambda, the more important the
                disagreement between the weights is. The lower lambda, the more important their log
                probabilities. Defaults to 0.01.

    Returns:
        float: the disagreement value (always nonnegative)
        
    Raises:
        AssertionError: if :py:attr:`weights` and :py:attr:`logprobs` have mismatching number of elements.
    """
    assert(len(weights) == len(logprobs) == 2), 'disagreement acquisition function works only with pairwise comparison queries, i.e., K must be 2.'

    kwargs.setdefault('lambda', 1e-2)
    term1 = np.prod(np.exp(logprobs))
    term2 = kwargs['lambda'] * np.linalg.norm(weights[0] - weights[1])
    return term1 + term2
    
    
def regret(weights: np.array, logprobs: List[float], planned_trajectories: List[Trajectory], **kwargs) -> float:
    """
    This function returns the regret value between two sets of reward weights (weights's).
    This is useful as an acquisition function when a trajectory planner is available and when
    the desired query contains only two trajectories. The pair of weights with the highest
    regret is found and then the best trajectories according to them forms the optimized query.
    
    This is implemented based on the following paper:
        - `Active Preference Learning using Maximum Regret <https://arxiv.org/abs/2005.04067>`_
    
    :TODO: This acquisition function requires all rewards to be positive, but there is no check
        for that.
    
    Args:
        weights (numpy.array): 2 x d array where each row is a set of reward weights. The
            regret between these two weights will be calculated.
        logprobs (List[float]): log probabilities of the given reward weights under the belief.
        planned_trajectories (List[Trajectory]): the optimal trajectories under the given reward weights.
        **kwargs: none used currently

    Returns:
        float: the regret value
        
    Raises:
        AssertionError: if :py:attr:`weights`, :py:attr:`logprobs` and :py:attr:`planned_trajectories`
            have mismatching number of elements.
    """
    assert(len(weights) == len(logprobs) == len(planned_trajectories) == 2), 'regret acquisition function works only with pairwise comparison queries, i.e., K must be 2.'

    term1 = np.prod(np.exp(logprobs))
    term2 = np.dot(weights[0], planned_trajectories[0].features) / np.dot(weights[1], planned_trajectories[0].features)
    term3 = np.dot(weights[1], planned_trajectories[1].features) / np.dot(weights[0], planned_trajectories[1].features)
    return term1 * (term2 + term3)

def thompson():
    """
    This function does nothing, but is added so that :py:mod:`aprel.querying.query_optimizer` can use it as a check.
    """
    pass # query optimizer handles the thompson sampling based querying

def random():
    """
    This function does nothing, but is added so that :py:mod:`aprel.querying.query_optimizer` can use it as a check.
    """
    pass # query optimizer handles the random querying -- it is computationally more efficient