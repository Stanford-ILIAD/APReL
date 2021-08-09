from typing import List, Dict
import numpy as np

from basics import Trajectory
from learning import Belief, TrueBelief
from learning import Query, PreferenceQuery, WeakComparisonQuery, FullRankingQuery

def mutual_information(belief: Belief, query: Query, **kwargs) -> float:
    if isinstance(belief, TrueBelief):
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
    if isinstance(belief, TrueBelief):
        user = belief.user_model.copy()
        probs = []
        for sample in belief.samples:
            user.params = sample
            probs.append(user.response_probabilities(query))
        M = len(probs)
        return np.sum((np.sum(probs, axis=0) / M) ** 2)
        
    raise NotImplementedError


def disagreement(omegas: np.array, logprobs: List[float], **kwargs) -> float:
    assert(len(omegas) == len(logprobs) == 2), 'disagreement acquisition function works only with pairwise comparison queries, i.e., K must be 2.'

    kwargs.setdefault('lambda', 1e-2)
    term1 = np.prod(np.exp(logprobs))
    term2 = kwargs['lambda'] * np.linalg.norm(omegas[0] - omegas[1])
    return term1 + term2
    
    
def regret(omegas: np.array, logprobs: List[float], planned_trajectories: List[Trajectory], **kwargs) -> float:
    assert(len(omegas) == len(logprobs) == len(planned_trajectories) == 2), 'regret acquisition function works only with pairwise comparison queries, i.e., K must be 2.'

    term1 = np.prod(np.exp(logprobs))
    term2 = np.dot(omegas[0], planned_trajectories[0].features) / np.dot(omegas[1], planned_trajectories[0].features)
    term3 = np.dot(omegas[1], planned_trajectories[1].features) / np.dot(omegas[0], planned_trajectories[1].features)
    return term1 * (term2 + term3)

def thompson():
    pass # query optimizer handles the thompson sampling based querying

def random():
    pass # query optimizer handles the random querying -- it is computationally more efficient