from typing import Callable, Dict, List, Tuple
import numpy as np

from learning import QueryWithResponse
from learning import User


class Sampler:
    def __init__(self, logprior: Callable):
        self.logprior = logprior
        
    def sample(self,
               user: User,
               dataset: List[QueryWithResponse],
               initial_point: Dict,
               proposal_distribution: Callable,
               num_samples: int,
               burnin: int = 200,
               thin: int = 20) -> Tuple[List[Dict], List[float]]:
        """Samples num_samples many reward weights from the posterior with the given dataset using Metropolis-Hastings.
        
        Args:
            user: the user model that is being learned.
            dataset: list of collectable data that come from the user.
            initial_point: initial point to start the chain for Metropolis-Hastings.
            proposal_distribution: proposal distribution for Metropolis-Hastings.
            num_samples: number of required samples.
            burnin: the first burnin samples are dropped to reduce the correlation with the initial sample.
            thin: one in every thin samples are taken to reduce auto-correlation between the samples.
        
        Returns:
            samples: list of dictionaries where each dictionary is a sample of user parameters.
            logprobs: list of float values where each entry is the log-probability of the corresponding sample.
        """
        
        samples = []
        logprobs = []
        curr_point = initial_point.copy()
        sampling_user = user.copy()
        sampling_user.params = curr_point
        curr_logprob = self.logprior(curr_point) + sampling_user.loglikelihood_dataset(dataset)
        samples.append(curr_point)
        logprobs.append(curr_logprob)
        accepts = 0
        for _ in range(burnin + thin * num_samples - 1):
            next_point = proposal_distribution(curr_point)
            sampling_user.params = next_point
            next_logprob = self.logprior(next_point) + sampling_user.loglikelihood_dataset(dataset)
            
            if np.log(np.random.rand()) < next_logprob - curr_logprob:
                curr_point = next_point.copy()
                curr_logprob = next_logprob
                accepts += 1

            samples.append(curr_point)
            logprobs.append(curr_logprob)
        return samples[burnin::thin], logprobs[burnin::thin]

