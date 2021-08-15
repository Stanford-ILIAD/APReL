"""
This file contains Belief classes, which store and update the belief distributions about the user whose reward function is being learned.

:TODO: GaussianBelief class will be implemented so that the library will include the following work:
    E. Biyik, N. Huynh, M. J. Kochenderger, D. Sadigh; "Active Preference-Based Gaussian Process Regression for Reward Learning", RSS'20.
"""
from typing import Callable, Dict, List, Tuple, Union
import numpy as np

from aprel.learning import User, QueryWithResponse
from aprel.utils import gaussian_proposal, uniform_logprior


class Belief:
    """An abstract class for Belief distributions."""
    def __init__(self):
        pass
        
    def update(self, data: Union[QueryWithResponse, List[QueryWithResponse]], **kwargs):
        """Updates the belief distribution with a given feedback or a list of feedbacks."""
        raise NotImplementedError


class LinearRewardBelief(Belief):
    """An abstract class for Belief distributions for the problems where reward function is assumed to be a linear function of the features."""
    def __init__(self):
        pass

    @property
    def mean(self) -> Dict:
        """Returns the mean parameters with respect to the belief distribution."""
        raise NotImplementedError


class SamplingBasedBelief(LinearRewardBelief):
    """
    A class for sampling based belief distributions.
    
    In this model, the entire dataset of user feedback is stored and used for calculating the true posterior value for any given set of
    parameters. A set of parameter samples are then sampled from this true posterior using Metropolis-Hastings algorithm.
    
    Parameters:
        logprior (Callable): The logarithm of the prior distribution over the user parameters.
        user_model (User): The user response model that will be assumed by this belief distribution.
        dataset (List[QueryWithResponse]): A list of user feeedbacks.
        initial_point (Dict): An initial set of user parameters for Metropolis-Hastings to start.
        logprior (Callable): The logarithm of the prior distribution over the user parameters. Defaults to a uniform distribution over the hyperball.
        num_samples (int): The number of parameter samples that will be sampled using Metropolis-Hastings.
        **kwargs: Hyperparameters for Metropolis-Hastings, which include:
            
            - `burnin` (int): The number of initial samples that will be discarded to remove the correlation with the initial parameter set.
            - `thin` (int): Once in every `thin` sample will be kept to reduce the autocorrelation between the samples.
            - `proposal_distribution` (Callable): The proposal distribution for the steps in Metropolis-Hastings.
            
    Attributes:
        user_model (User): The user response model that is assumed by the belief distribution.
        dataset (List[QueryWithResponse]): A list of user feeedbacks.
        num_samples (int): The number of parameter samples that will be sampled using Metropolis-Hastings.
        sampling_params (Dict): Hyperparameters for Metropolis-Hastings, which include:
        
            - `burnin` (int): The number of initial samples that will be discarded to remove the correlation with the initial parameter set.
            - `thin` (int): Once in every `thin` sample will be kept to reduce the autocorrelation between the samples.
            - `proposal_distribution` (Callable): The proposal distribution for the steps in Metropolis-Hastings.
    """
    def __init__(self,
                 user_model: User,
                 dataset: List[QueryWithResponse],
                 initial_point: Dict,
                 logprior: Callable = uniform_logprior,
                 num_samples: int = 100,
                 **kwargs):
        super(SamplingBasedBelief, self).__init__()
        self.logprior = logprior
        self.user_model = user_model
        self.dataset = []
        self.num_samples = num_samples
        
        kwargs.setdefault('burnin', 200)
        kwargs.setdefault('thin', 20)
        kwargs.setdefault('proposal_distribution', gaussian_proposal)
        self.sampling_params = kwargs
        self.update(dataset, initial_point)
        
    def update(self, data: Union[QueryWithResponse, List[QueryWithResponse]], initial_point: Dict = None):
        """
        Updates the belief distribution based on the new feedback (query-response pairs), by adding these to
        the current dataset and then re-sampling with Metropolis-Hastings.
        Args:
            data (QueryWithResponse or List[QueryWithResponse]): one or more QueryWithResponse, which
                contains multiple trajectory options and the index of the one the user selected as most optimal
            initial_point (Dict): the initial point to start Metropolis-Hastings from, will be set to the mean
                           from the previous distribution if None
        """
        if isinstance(data, list):
            self.dataset.extend(data)
        else:
            self.dataset.append(data)
        if initial_point is None:
            initial_point = self.mean
        
        self.create_samples(initial_point)
        
        
    def create_samples(self, initial_point: Dict) -> Tuple[List[Dict], List[float]]:
        """Samples num_samples many user parameters from the posterior using Metropolis-Hastings.
        
        Args:
            initial_point (Dict): initial point to start the chain for Metropolis-Hastings.
        
        Returns:
            2-tuple:
                - List[Dict]: dictionaries where each dictionary is a sample of user parameters.
                - List[float]: float values where each entry is the log-probability of the corresponding sample.
        """
        burnin = self.sampling_params['burnin']
        thin = self.sampling_params['thin']
        proposal_distribution = self.sampling_params['proposal_distribution']
        
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
    def mean(self) -> Dict:
        """Returns the mean of the belief distribution by taking the mean over the samples generated by Metropolis-Hastings."""
        mean_params = {}
        for key in self.samples[0].keys():
            mean_params[key] = np.mean([self.samples[i][key] for i in range(self.num_samples)], axis=0)
            if key == 'weights':
                mean_params[key] /= np.linalg.norm(mean_params[key])
        return mean_params
