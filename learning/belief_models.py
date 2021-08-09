from typing import Callable, Dict, List
import numpy as np

from learning import User, Sampler, QueryWithResponse

# TODO: GaussianBelief class will be implemented so that the library will include the following work:
# E. Biyik, N. Huynh, M. J. Kochenderger, D. Sadigh; "Active Preference-Based Gaussian Process Regression for Reward Learning", RSS'20.
# Note: this may also require changes in the user_models.py, because GPs are non-parametric, i.e., there is no omega.

class Belief:
    def __init__(self):
        pass
        
    @property
    def mean(self):
        raise NotImplementedError


class TrueBelief(Belief):
    def __init__(self,
                 logprior: Callable,
                 user_model: User,
                 dataset: List[QueryWithResponse],
                 initial_point: Dict,
                 proposal_distribution: Callable,
                 num_samples: int = 100,
                 burnin: int = 200,
                 thin: int = 20):
        """Initializes a true belief object given a dataset of user feedback.
        
        The true belief model operates based on the samples from the true belief distribution.
        
        Args:
            dataset: a list of data samples where each sample comes from the user
        """
        super(TrueBelief, self).__init__()
        self.user_model = user_model
        self.num_samples = num_samples
        sampler = Sampler(logprior)
        self.samples, self.logprobs = sampler.sample(user_model, dataset, initial_point, proposal_distribution, num_samples, burnin, thin)

    @property
    def mean(self):
        mean_params = {}
        for key in self.samples[0].keys():
            mean_params[key] = np.mean([self.samples[i][key] for i in range(self.num_samples)], axis=0)
            if key == 'omega':
                mean_params[key] /= np.linalg.norm(mean_params[key])
        return mean_params
