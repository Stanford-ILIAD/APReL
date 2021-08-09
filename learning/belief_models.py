from typing import Callable, Dict, List, Tuple, Union
import numpy as np

from learning import User, QueryWithResponse
from utils import gaussian_proposal

# TODO: GaussianBelief class will be implemented so that the library will include the following work:
# E. Biyik, N. Huynh, M. J. Kochenderger, D. Sadigh; "Active Preference-Based Gaussian Process Regression for Reward Learning", RSS'20.
# Note: this may also require changes in the user_models.py, because GPs are non-parametric, i.e., there is no omega.

class Belief:
    def __init__(self):
        pass
        
    @property
    def mean(self):
        raise NotImplementedError
        
    def update(self, data: Union[QueryWithResponse, List[QueryWithResponse]], **kwargs):
        raise NotImplementedError


class TrueBelief(Belief):
    def __init__(self,
                 logprior: Callable,
                 user_model: User,
                 dataset: List[QueryWithResponse],
                 initial_point: Dict,
                 num_samples: int = 100,
                 **kwargs):
        """Initializes a true belief object given a dataset of user feedback.
        
        The true belief model operates based on the samples from the true belief distribution.
        
        Args:
            logprior: log of the prior belief distribution.
            user_model: the user model that is being learned.
            dataset: list of collectable data that come from the user.
            initial_point: initial point to start the chain for Metropolis-Hastings.
            proposal_distribution: proposal distribution for Metropolis-Hastings.
            num_samples: number of required samples.
            **kwargs: sampling-related parameters:
                      proposal_distribution, burning and thin for the current sampling algorithm.
        """
        super(TrueBelief, self).__init__()
        self.logprior = logprior
        self.user_model = user_model
        self.dataset = []
        self.num_samples = num_samples
        
        kwargs.setdefault('burnin', 200)
        kwargs.setdefault('thin', 20)
        kwargs.setdefault('proposal_distribution', gaussian_proposal)
        self.optimization_params = kwargs
        self.update(dataset, initial_point)
        
    def update(self, data: Union[QueryWithResponse, List[QueryWithResponse]], initial_point: Dict = None):
        if isinstance(data, list):
            self.dataset.extend(data)
        else:
            self.dataset.append(data)
        if initial_point is None:
            initial_point = self.mean
        
        self.create_samples(initial_point)
        
        
    def create_samples(self, initial_point: Dict) -> Tuple[List[Dict], List[float]]:
        """Samples num_samples many reward weights from the posterior using Metropolis-Hastings.
        
        Args:
            initial_point: initial point to start the chain for Metropolis-Hastings.
        
        Returns:
            samples: list of dictionaries where each dictionary is a sample of user parameters.
            logprobs: list of float values where each entry is the log-probability of the corresponding sample.
        """
        burnin = self.optimization_params['burnin']
        thin = self.optimization_params['thin']
        proposal_distribution = self.optimization_params['proposal_distribution']
        
        samples = []
        logprobs = []
        curr_point = initial_point.copy()
        sampling_user = self.user_model.copy()
        sampling_user.params = curr_point
        curr_logprob = self.logprior(curr_point) + sampling_user.loglikelihood_dataset(self.dataset)
        samples.append(curr_point)
        logprobs.append(curr_logprob)
        for _ in range(burnin + thin * self.num_samples - 1):
            next_point = proposal_distribution(curr_point)
            sampling_user.params = next_point
            next_logprob = self.logprior(next_point) + sampling_user.loglikelihood_dataset(self.dataset)
            if np.log(np.random.rand()) < next_logprob - curr_logprob:
                curr_point = next_point.copy()
                curr_logprob = next_logprob
            samples.append(curr_point)
            logprobs.append(curr_logprob)
        self.samples, self.logprobs = samples[burnin::thin], logprobs[burnin::thin]

    @property
    def mean(self):
        mean_params = {}
        for key in self.samples[0].keys():
            mean_params[key] = np.mean([self.samples[i][key] for i in range(self.num_samples)], axis=0)
            if key == 'omega':
                mean_params[key] /= np.linalg.norm(mean_params[key])
        return mean_params
