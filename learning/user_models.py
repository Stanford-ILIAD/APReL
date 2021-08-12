"""Classes to model the expert user who is answering the queries."""
from typing import Dict, List, Union
import numpy as np
import scipy.special as ssp
from copy import deepcopy

from learning import Query, PreferenceQuery, WeakComparisonQuery, FullRankingQuery
from learning import QueryWithResponse, Demonstration, Preference, WeakComparison, FullRanking

class User:
    """An abstract class to model the user, with functions to return the probability of a given response to a query,
    or to generate a response to a query."""
    def __init__(self, params_dict: Dict = None):
        if params_dict is not None:
            self._params = params_dict.copy()
        else:
            self._params = {}
    
    @property
    def params(self):
        return self._params
        
    @params.setter
    def params(self, params_dict):
        params_dict_copy = params_dict.copy()
        for key, value in self._params.items():
            params_dict_copy.setdefault(key, value)
        self._params = params_dict_copy
        
    def copy(self):
        return deepcopy(self)
        
    def response_logprobabilities(self, query: Query) -> np.array:
        """Returns the log probability for each response in the response set for the query."""
        raise NotImplementedError
        
    def response_probabilities(self, query: Query) -> np.array:
        """Returns the probability for each response in the response set for the query."""
        return np.exp(self.response_logprobabilities(query))
        
    def loglikelihood(self, data: QueryWithResponse) -> float:
        """Returns the log probability of the given response to the query."""
        logprobs = self.response_logprobabilities(data)
        if isinstance(data, Preference) or isinstance(data, WeakComparison):
            idx = np.where(data.query.response_set == data.response)[0][0]
        elif isinstance(data, FullRanking):
            idx = np.where((data.query.response_set == data.response).all(axis=1))[0][0]
        return logprobs[idx]

    def likelihood(self, data: QueryWithResponse) -> float:
        """Returns the probability of the given response to the query."""
        return np.exp(self.loglikelihood(data))
        
    def loglikelihood_dataset(self, dataset: List[QueryWithResponse]) -> float:
        """Returns the (unnormalized) loglikelihood for the dataset under the conditional independence assumption."""
        return np.sum([self.loglikelihood(data) for data in dataset])
        
    def likelihood_dataset(self, dataset: List[QueryWithResponse]) -> float:
        """Returns the (unnormalized) likelihood for the dataset under the conditional independence assumption."""
        return np.exp(self.loglikelihood_dataset(dataset))

    def respond(self, queries: Union[Query, List[Query]]) -> List:
        """Simulates the user's response to the given query."""
        if not isinstance(queries, list):
            queries = [queries]
        responses = []
        for query in queries:
            probs = self.response_probabilities(query)
            idx = np.random.choice(len(probs), p=probs)
            responses.append(query.response_set[idx])
        return responses


class SoftmaxUser(User):
    def __init__(self, params_dict: Dict):
        """Initializes a softmax user object.
        
        Args:
            params_dict: the parameters of the softmax user model, which are:
                omega,  the weights of the linear reward function;
                beta,   rationality coefficient for comparisons and rankings;
                beta_D, rationality coefficient for demonstrations;
                delta,  the perceivable difference parameter for weak comparison queries.
        """
        assert('omega' in params_dict), 'omega is a required parameter for the softmax user model.'       
        params_dict_copy = params_dict.copy()
        params_dict_copy.setdefault('beta', 1.0)
        params_dict_copy.setdefault('beta_D', 1.0)
        params_dict_copy.setdefault('delta', 0.1)
        
        super(SoftmaxUser, self).__init__(params_dict_copy)
        
    def response_logprobabilities(self, query: Query) -> np.array:
        """Returns the response logprobabilities for each possible respond."""
        if isinstance(query, PreferenceQuery):
            rewards = self.params['beta'] * np.dot(query.slate.features_matrix, self.params['omega'])
            return rewards - ssp.logsumexp(rewards)
            
        elif isinstance(query, WeakComparisonQuery):
            rewards = self.params['beta'] * np.dot(query.slate.features_matrix, self.params['omega'])
            logprobs = np.zeros((3))
            logprobs[1] = -np.log(1 + np.exp(self.params['delta'] + rewards[1] - rewards[0]))
            logprobs[2] = -np.log(1 + np.exp(self.params['delta'] + rewards[0] - rewards[1]))
            logprobs[0] = np.log(np.exp(2*self.params['delta']) - 1) + logprobs[1] + logprobs[2]
            return logprobs
            
        elif isinstance(query, FullRankingQuery):
            rewards = self.params['beta'] * np.dot(query.slate.features_matrix, self.params['omega'])
            logprobs = np.zeros(len(query.response_set))
            for response_id in range(len(query.response_set)):
                response = query.response_set[response_id]
                sorted_rewards = rewards[response]
                logprobs[response_id] = np.sum([sorted_rewards[i] - ssp.logsumexp(sorted_rewards[i:]) for i in range(len(response))])
            return logprobs

    def loglikelihood(self, data: QueryWithResponse) -> float:
        """Returns the loglikelihood for the data. Not needed for all data types, but overrides the parent class for faster execution.
        
        The output value is log(unnormalized likelihood) if the data is a demonstration.
        Otherwise it is the true loglikelihood.
        """
        if isinstance(data, Demonstration):
            return self.params['beta_D'] * np.dot(data.features, self.params['omega'])
        
        elif isinstance(data, Preference):
            rewards = self.params['beta'] * np.dot(data.query.slate.features_matrix, self.params['omega'])
            return rewards[data.response] - ssp.logsumexp(rewards)
            
        elif isinstance(data, WeakComparison):
            rewards = self.params['beta'] * np.dot(data.query.slate.features_matrix, self.params['omega'])
            
            logp0 = -np.log(1 + np.exp(self.params['delta'] + rewards[1] - rewards[0]))
            if data.response == 0: return logp0
            
            logp1 = -np.log(1 + np.exp(self.params['delta'] + rewards[0] - rewards[1]))
            if data.response == 1: return logp1
            
            if data.response == -1:
                return np.log(np.exp(2*self.params['delta']) - 1) + logp0 + logp1
                
        elif isinstance(data, FullRanking):
            rewards = self.params['beta'] * np.dot(data.query.slate.features_matrix, self.params['omega'])
            sorted_rewards = rewards[data.response]
            return np.sum([sorted_rewards[i] - ssp.logsumexp(sorted_rewards[i:]) for i in range(len(data.response))])
            
        raise NotImplementedError("User response model for the given data is not implemented.")


class HumanUser(User):
    def __init__(self):
        """Initializes a human user object."""
        super(HumanUser, self).__init__()
        
    def respond(self, queries: Union[Query, List[Query]]) -> List:
        """Ask the queries to the user."""
        if not isinstance(queries, list):
            queries = [queries]
        responses = []
        for query in queries:
            responses.append(query.visualize())
        return responses
