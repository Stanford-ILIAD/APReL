"""
This file contains classes which have functions to optimize the queries to ask the human.
"""
from typing import Callable, List, Tuple
import itertools
import numpy as np
from scipy.spatial import ConvexHull
import warnings

from aprel.basics import Trajectory, TrajectorySet
from aprel.learning import Belief, SamplingBasedBelief, User, SoftmaxUser
from aprel.learning import Query, PreferenceQuery, WeakComparisonQuery, FullRankingQuery
from aprel.querying import mutual_information, volume_removal, disagreement, regret, random, thompson
from aprel.utils import kMedoids, dpp_mode, default_query_distance


class QueryOptimizer:
    """
    An abstract class for query optimizer frameworks.
    
    Attributes:
        acquisition_functions (Dict): keeps name-function pairs for the acquisition functions. If new acquisition
            functions are implemented, they should be added to this dictionary.
    """
    def __init__(self):
        self.acquisition_functions = {'mutual_information': mutual_information,
                                      'volume_removal': volume_removal,
                                      'disagreement': disagreement,
                                      'regret': regret,
                                      'random': random,
                                      'thompson': thompson}


class QueryOptimizerDiscreteTrajectorySet(QueryOptimizer):
    """
    Query optimization framework that assumes a discrete set of trajectories is available. The query optimization
    is then performed over this discrete set.
    
    Parameters:
        trajectory_set (TrajectorySet): The set of trajectories from which the queries will be optimized. This set
            defines the possible set of trajectories that may show up in the optimized query.

    Attributes:
        trajectory_set (TrajectorySet): The set of trajectories from which the queries are optimized. This set
            defines the possible set of trajectories that may show up in the optimized query.
    """
    def __init__(self, trajectory_set: TrajectorySet):
        super(QueryOptimizerDiscreteTrajectorySet, self).__init__()
        self.trajectory_set = trajectory_set
        
    def argplanner(self, user: User) -> int:
        """
        Given a user model, returns the index of the trajectory that best fits the user in the trajectory set.
        
        Args:
            user (User): The user object for whom the optimal trajectory is being searched.
            
        Returns:
            int: The index of the optimal trajectory in the trajectory set.
        """
        if isinstance(user, SoftmaxUser):
            return np.asscalar(np.argmax(user.reward(self.trajectory_set)))
        raise NotImplementedError("The planner has not been implemented for the given user model.")

    def planner(self, user: User) -> Trajectory:
        """
        Given a user model, returns the trajectory in the trajectory set that best fits the user.
        
        Args:
            user (User): The user object for whom the optimal trajectory is being searched.
            
        Returns:
            Trajectory: The optimal trajectory in the trajectory set.
        """
        return self.trajectory_set[self.argplanner(user)]
        
    def optimize(self,
                 acquisition_func_str: str,
                 belief: Belief,
                 initial_query: Query,
                 batch_size: int = 1,
                 optimization_method: str = 'exhaustive_search',
                 **kwargs) -> Tuple[List[Query], np.array]:
        """
        This function generates the optimal query or the batch of queries to ask to the user given a belief
        distribution about them. It also returns the acquisition function values of the optimized queries.
        
        Args:
            acquisition_func_str (str): the name of the acquisition function used to decide the value of each query.
                Currently implemented options are:
                
                - `disagreement`: Based on `Katz. et al. (2019) <https://arxiv.org/abs/1907.05575>`_.
                - `mutual_information`: Based on `Bıyık et al. (2019) <https://arxiv.org/abs/1910.04365>`_.
                - `random`: Randomly chooses a query.
                - `regret`: Based on `Wilde et al. (2020) <https://arxiv.org/abs/2005.04067>`_.
                - `thompson`: Based on `Tucker et al. (2019) <https://arxiv.org/abs/1909.12316>`_.
                - `volume_removal`: Based on `Sadigh et al. (2017) <http://m.roboticsproceedings.org/rss13/p53.pdf>`_ and `Bıyık et al. <https://arxiv.org/abs/1904.02209>`_.
            belief (Belief): the current belief distribution over the user.
            initial_query (Query): an initial query such that the output query will have the same type.
            batch_size (int): the number of queries to return.
            optimization_method (str): the name of the method used to select queries. Currently implemented options are:
            
                - `exhaustive_search`: Used for exhaustively searching a single query.
                - `greedy`: Exhaustively searches for the top :py:attr:`batch_size` queries in terms of the acquisition function.
                - `medoids`: Batch generation method based on `Bıyık et al. (2018) <https://arxiv.org/abs/1810.04303>`_.
                - `boundary_medoids`: Batch generation method based on `Bıyık et al. (2018) <https://arxiv.org/abs/1810.04303>`_.
                - `successive_elimination`: Batch generation method based on `Bıyık et al. (2018) <https://arxiv.org/abs/1810.04303>`_.
                - `dpp`: Batch generation method based on `Bıyık et al. (2019) <https://arxiv.org/abs/1906.07975>`_.
            **kwargs: extra arguments needed for specific optimization methods or acquisition functions.
            
            Returns:
                2-tuple:
                
                    - List[Query]: The list of optimized queries. **Note**: Even if :py:attr:`batch_size` is 1, a list is returned.
                    - numpy.array: An array of floats that keep the acquisition function values corresponding to the output queries.
        """
        assert(acquisition_func_str in self.acquisition_functions), 'Unknown acquisition function.'
        acquisition_func = self.acquisition_functions[acquisition_func_str]
        
        assert(batch_size > 0 and isinstance(batch_size, int)), 'Invalid batch_size ' + str(batch_size)
        assert(optimization_method in ['exhaustive_search', 'greedy', 'medoids', 'boundary_medoids', 'successive_elimination', 'dpp']), 'Unknown optimization_method ' + str(optimization_method)
        if batch_size > 1 and optimization_method == 'exhaustive_search':
            warnings.warn('Since batch size > 1, ignoring exhaustive search and using greedy batch selection instead.')
            optimization_method = 'greedy'
        elif batch_size == 1 and optimization_method in ['greedy', 'medoids', 'boundary_medoids', 'successive_elimination', 'dpp']:
            warnings.warn('Since batch size == 1, ignoring the batch selection method and using exhaustive search instead.')
            optimization_method = 'exhaustive_search'
                 
        if optimization_method == 'exhaustive_search':
            return self.exhaustive_search(acquisition_func, belief, initial_query, **kwargs)
        elif optimization_method == 'greedy':
            return self.greedy_batch(acquisition_func, belief, initial_query, batch_size, **kwargs)
        elif optimization_method == 'medoids':
            return self.medoids_batch(acquisition_func, belief, initial_query, batch_size, **kwargs)
        elif optimization_method == 'boundary_medoids':
            return self.boundary_medoids_batch(acquisition_func, belief, initial_query, batch_size, **kwargs)
        elif optimization_method == 'successive_elimination':
            return self.successive_elimination_batch(acquisition_func, belief, initial_query, batch_size, **kwargs)
        elif optimization_method == 'dpp':
            return self.dpp_batch(acquisition_func, belief, initial_query, batch_size, **kwargs)
            
        raise NotImplementedError('unknown optimization method for QueryOptimizerDiscreteTrajectorySet: ' + optimization_method + '.')
        
    def exhaustive_search(self,
                          acquisition_func: Callable,
                          belief: Belief,
                          initial_query: Query,
                          **kwargs) -> Tuple[List[Query], np.array]:
        """
        Searches over the possible queries to find the singular most optimal query.
        
        Args:
            acquisition_func (Callable): the acquisition function to be maximized.
            belief (Belief): the current belief distribution over the user.
            initial_query (Query): an initial query such that the output query will have the same type.
            **kwargs: extra arguments needed for specific acquisition functions.

        Returns:
                2-tuple:
                
                    - List[Query]: The optimal query as a list of one :class:`.Query`.
                    - numpy.array: An array of floats that keep the acquisition function value corresponding to the output query.
        """
        return self.greedy_batch(acquisition_func, belief, initial_query, batch_size=1, **kwargs)

    def greedy_batch(self,
                     acquisition_func: Callable,
                     belief: Belief,
                     initial_query: Query,
                     batch_size: int,
                     **kwargs) -> Tuple[List[Query], np.array]:
        """
        Uses the greedy method to find a batch of queries by selecting the :py:attr:`batch_size` individually most optimal queries.
        
        Args:
            acquisition_func (Callable): the acquisition function to be maximized by each individual query.
            belief (Belief): the current belief distribution over the user.
            initial_query (Query): an initial query such that the output query will have the same type.
            batch_size (int): the batch size of the output.
            **kwargs: extra arguments needed for specific acquisition functions.

        Returns:
                2-tuple:
                
                    - List[Query]: The optimized batch of queries as a list.
                    - numpy.array: An array of floats that keep the acquisition function values corresponding to the output queries.
        """
        if isinstance(initial_query, PreferenceQuery) or isinstance(initial_query, WeakComparisonQuery) or isinstance(initial_query, FullRankingQuery):
            if acquisition_func is random:
                best_batch = [initial_query.copy() for _ in range(batch_size)]
                for i in range(batch_size):
                    best_batch[i].slate = self.trajectory_set[np.random.choice(self.trajectory_set.size, size=initial_query.K, replace=False)]
                return best_batch, np.array([1. for _ in range(batch_size)])
                
            elif acquisition_func is thompson and isinstance(belief, SamplingBasedBelief):
                subsets = np.array([list(tup) for tup in itertools.combinations(np.arange(belief.num_samples), initial_query.K)])
                if len(subsets) < batch_size:
                    batch_size = len(subsets)
                    warnings.warn('The number of possible queries is smaller than the batch size. Automatically reducing the batch size.')
                temp_user = belief.user_model.copy()
                planned_traj_ids = []
                for sample in belief.samples:
                    temp_user.params = sample
                    planned_traj_ids.append(self.argplanner(temp_user))
                belief_logprobs = np.array(belief.logprobs)
                best_batch = [initial_query.copy() for _ in range(batch_size)]
                
                unique_traj_ids, inverse = np.unique(planned_traj_ids, return_inverse=True)
                if len(unique_traj_ids) < initial_query.K:
                    remaining_ids = np.setdiff1d(np.arange(self.trajectory_set.size), unique_traj_ids)
                    missing_count = initial_query.K - len(unique_traj_ids)
                    for i in range(batch_size):
                        ids = np.append(unique_traj_ids, np.random.choice(remaining_ids, size=missing_count, replace=False))
                        best_batch[i].slate = self.trajectory_set[ids]
                else:
                    unique_probs = np.array([np.exp(belief_logprobs[inverse==i]).sum() for i in range(len(unique_traj_ids))])
                    if np.isclose(unique_probs.sum(), 0):
                        unique_probs = np.ones_like(unique_probs)
                    unique_probs /= unique_probs.sum()
                    for i in range(batch_size):
                        best_batch[i].slate = self.trajectory_set[np.random.choice(unique_traj_ids, size=initial_query.K, replace=False, p=unique_probs)]
                return best_batch, np.array([1. for _ in range(batch_size)])

            elif acquisition_func is mutual_information or acquisition_func is volume_removal:
                subsets = np.array([list(tup) for tup in itertools.combinations(np.arange(self.trajectory_set.size), initial_query.K)])
                if len(subsets) < batch_size:
                    batch_size = len(subsets)
                    warnings.warn('The number of possible queries is smaller than the batch size. Automatically reducing the batch size.')
                vals = []
                for ids in subsets:
                    curr_query = initial_query.copy()
                    curr_query.slate = self.trajectory_set[ids]
                    vals.append(acquisition_func(belief, curr_query, **kwargs))
                vals = np.array(vals)
                inds = np.argpartition(vals, -batch_size)[-batch_size:]
                
                best_batch = [initial_query.copy() for _ in range(batch_size)]
                for i in range(batch_size):
                    best_batch[i].slate = self.trajectory_set[subsets[inds[i]]]
                return best_batch, vals[inds]
                
            elif acquisition_func is disagreement and isinstance(belief, SamplingBasedBelief):
                assert(initial_query.K == 2), 'disagreement acquisition function works only with pairwise comparison queries, i.e., K must be 2.'
                subsets = np.array([list(tup) for tup in itertools.combinations(np.arange(belief.num_samples), initial_query.K)])
                if len(subsets) < batch_size:
                    batch_size = len(subsets)
                    warnings.warn('The number of possible queries is smaller than the batch size. Automatically reducing the batch size.')
                vals = []
                belief_samples = np.array(belief.samples)
                belief_logprobs = np.array(belief.logprobs)
                for ids in subsets:
                    weights = np.array([sample['weights'] for sample in belief_samples[ids]])
                    vals.append(acquisition_func(weights, belief_logprobs[ids], **kwargs))
                vals = np.array(vals)
                inds = np.argpartition(vals, -batch_size)[-batch_size:]

                best_batch = [initial_query.copy() for _ in range(batch_size)]
                temp_user = belief.user_model.copy()
                for i in range(batch_size):
                    trajectories = []
                    for best_id in subsets[inds[i]]:
                        temp_user.params = belief.samples[best_id]
                        trajectories.append(self.planner(temp_user))
                    best_batch[i].slate = TrajectorySet(trajectories)
                return best_batch, vals[inds]
                
            elif acquisition_func is regret and isinstance(belief, SamplingBasedBelief):
                assert(initial_query.K == 2), 'regret acquisition function works only with pairwise comparison queries, i.e., K must be 2.'
                subsets = np.array([list(tup) for tup in itertools.combinations(np.arange(belief.num_samples), initial_query.K)])
                if len(subsets) < batch_size:
                    batch_size = len(subsets)
                    warnings.warn('The number of possible queries is smaller than the batch size. Automatically reducing the batch size.')
                temp_user = belief.user_model.copy()
                trajectories = []
                for sample in belief.samples:
                    temp_user.params = sample
                    trajectories.append(self.planner(temp_user))
                planned_trajs = TrajectorySet(trajectories)
                vals = []
                belief_samples = np.array(belief.samples)
                belief_logprobs = np.array(belief.logprobs)
                for ids in subsets:
                    weights = np.array([sample['weights'] for sample in belief_samples[ids]])
                    vals.append(acquisition_func(weights, belief_logprobs[ids], planned_trajs[ids], **kwargs))
                vals = np.array(vals)
                inds = np.argpartition(vals, -batch_size)[-batch_size:]
                
                best_batch = [initial_query.copy() for _ in range(batch_size)]
                for i in range(batch_size):
                    best_batch[i].slate = planned_trajs[subsets[inds[i]]]
                return best_batch, vals[inds]
            
        raise NotImplementedError('greedy batch has not been implemented for the given query and belief types.')


    def medoids_batch(self,
                     acquisition_func: Callable,
                     belief: Belief,
                     initial_query: Query,
                     batch_size: int,
                     **kwargs) -> Tuple[List[Query], np.array]:
        """
        Uses the medoids method to find a batch of queries. See
        `Batch Active Preference-Based Learning of Reward Functions <https://arxiv.org/abs/1810.04303>`_ for
        more information about the method.
        
        Args:
            acquisition_func (Callable): the acquisition function to be maximized by each individual query.
            belief (Belief): the current belief distribution over the user.
            initial_query (Query): an initial query such that the output query will have the same type.
            batch_size (int): the batch size of the output.
            **kwargs: Hyperparameters `reduced_size`, `distance`, and extra arguments needed for specific acquisition functions.
                
                - `reduced_size` (int): The hyperparameter `B` in the original method. This method first greedily chooses `B` queries from the feasible set of queries out of the trajectory set, and then applies the medoids selection. Defaults to 100.
                - `distance` (Callable): A distance function which returns a pairwise distance matrix (numpy.array) when inputted a list of queries. Defaults to :py:meth:`aprel.utils.batch_utils.default_query_distance`.

        Returns:
                2-tuple:
                
                    - List[Query]: The optimized batch of queries as a list.
                    - numpy.array: An array of floats that keep the acquisition function values corresponding to the output queries.
        """
        kwargs.setdefault('reduced_size', 100)
        kwargs.setdefault('distance', default_query_distance)
        top_queries, vals = self.greedy_batch(acquisition_func, belief, initial_query, batch_size=kwargs['reduced_size'], **kwargs)
        del kwargs['reduced_size']
        distances = kwargs['distance'](top_queries, **kwargs)
        medoid_ids = kMedoids(distances, batch_size)
        if len(medoid_ids) < batch_size:
            # There were too many duplicate points, so we ended up with fewer medoids than we needed.
            remaining_ids = np.setdiff1d(np.arange(len(vals)), medoid_ids)
            remaining_vals = vals[remaining_ids]
            missing_count = batch_size - len(medoid_ids)
            ids_to_add = remaining_ids[np.argpartition(remaining_vals, -missing_count)[-missing_count:]]
            medoid_ids = np.concatenate((medoid_ids, ids_to_add))
        return [top_queries[idx] for idx in medoid_ids], vals[medoid_ids]
        
    def boundary_medoids_batch(self,
                               acquisition_func: Callable,
                               belief: Belief,
                               initial_query: Query,
                               batch_size: int,
                               **kwargs) -> Tuple[List[Query], np.array]:
        """
        Uses the boundary medoids method to find a batch of queries. See
        `Batch Active Preference-Based Learning of Reward Functions <https://arxiv.org/abs/1810.04303>`_ for
        more information about the method.
        
        Args:
            acquisition_func (Callable): the acquisition function to be maximized by each individual query.
            belief (Belief): the current belief distribution over the user.
            initial_query (Query): an initial query such that the output query will have the same type.
            batch_size (int): the batch size of the output.
            **kwargs: Hyperparameters `reduced_size`, `distance`, and extra arguments needed for specific acquisition functions.
                
                - `reduced_size` (int): The hyperparameter `B` in the original method. This method first greedily chooses `B` queries from the feasible set of queries out of the trajectory set, and then applies the boundary medoids selection. Defaults to 100.
                - `distance` (Callable): A distance function which returns a pairwise distance matrix (numpy.array) when inputted a list of queries. Defaults to :py:meth:`aprel.utils.batch_utils.default_query_distance`.

        Returns:
                2-tuple:
                
                    - List[Query]: The optimized batch of queries as a list.
                    - numpy.array: An array of floats that keep the acquisition function values corresponding to the output queries.
        """
        kwargs.setdefault('reduced_size', 100)
        kwargs.setdefault('distance', default_query_distance)
        top_queries, vals = self.greedy_batch(acquisition_func, belief, initial_query, batch_size=kwargs['reduced_size'], **kwargs)
        del kwargs['reduced_size']
        assert initial_query.K == 2, 'Boundary medoids batch selection method does not support large slates, use K = 2.'
        feature_dim = initial_query.slate.features_matrix.shape[1]
        if feature_dim > 7:
            warnings.warn('Feature dimensionality is too high: ' + str(feature_dim) + '. Boundary medoids might be too slow.')
        
        features_diff = [query.slate.features_matrix[0] - query.slate.features_matrix[1] for query in top_queries]
        hull = ConvexHull(features_diff)
        simplices = np.unique(hull.simplices)
        if len(simplices) < batch_size:
            # If boundary has fewer points than the batch, then fill it with greedy queries
            medoid_ids = simplices
            remaining_ids = np.setdiff1d(np.arange(len(vals)), medoid_ids)
            remaining_vals = vals[remaining_ids]
            missing_count = batch_size - len(simplices)
            ids_to_add = remaining_ids[np.argpartition(remaining_vals, -missing_count)[-missing_count:]]
            medoid_ids = np.concatenate((medoid_ids, ids_to_add))
        else:
            # Otherwise, select the medoids among the boundary queries
            distances = kwargs['distance']([top_queries[i] for i in simplices], **kwargs)
            temp_ids = kMedoids(distances, batch_size)
            medoid_ids = simplices[temp_ids]
        return [top_queries[idx] for idx in medoid_ids], vals[medoid_ids]

    def successive_elimination_batch(self,
                                     acquisition_func: Callable,
                                     belief: Belief,
                                     initial_query: Query,
                                     batch_size: int,
                                     **kwargs) -> Tuple[List[Query], np.array]:
        """
        Uses the successive elimination method to find a batch of queries. See
        `Batch Active Preference-Based Learning of Reward Functions <https://arxiv.org/abs/1810.04303>`_ for
        more information about the method.
        
        Args:
            acquisition_func (Callable): the acquisition function to be maximized by each individual query.
            belief (Belief): the current belief distribution over the user.
            initial_query (Query): an initial query such that the output query will have the same type.
            batch_size (int): the batch size of the output.
            **kwargs: Hyperparameters `reduced_size`, `distance`, and extra arguments needed for specific acquisition functions.
                
                - `reduced_size` (int): The hyperparameter `B` in the original method. This method first greedily chooses `B` queries from the feasible set of queries out of the trajectory set, and then applies the boundary medoids selection. Defaults to 100.
                - `distance` (Callable): A distance function which returns a pairwise distance matrix (numpy.array) when inputted a list of queries. Defaults to :py:meth:`aprel.utils.batch_utils.default_query_distance`.

        Returns:
                2-tuple:
                
                    - List[Query]: The optimized batch of queries as a list.
                    - numpy.array: An array of floats that keep the acquisition function values corresponding to the output queries.
        """
        kwargs.setdefault('reduced_size', 100)
        kwargs.setdefault('distance', default_query_distance)
        top_queries, vals = self.greedy_batch(acquisition_func, belief, initial_query, batch_size=kwargs['reduced_size'], **kwargs)
        del kwargs['reduced_size']
        distances = kwargs['distance'](top_queries, **kwargs)
        distances[np.isclose(distances, 0)] = np.inf
        while len(top_queries) > batch_size:
            ij_min = np.where(distances == np.min(distances))
            if len(ij_min) > 1 and len(ij_min[0]) > 1:
                ij_min = ij_min[0]
            elif len(ij_min) > 1:
                ij_min = np.array([ij_min[0],ij_min[1]])

            if vals[ij_min[0]] < vals[ij_min[1]]:
                delete_id = ij_min[1]
            else:
                delete_id = ij_min[0]
            distances = np.delete(distances, delete_id, axis=0)
            distances = np.delete(distances, delete_id, axis=1)
            vals = np.delete(vals, delete_id)
            top_queries = np.delete(top_queries, delete_id, axis=0)
        return list(top_queries), vals
        
    def dpp_batch(self,
                  acquisition_func: Callable,
                  belief: Belief,
                  initial_query: Query,
                  batch_size: int,
                  **kwargs) -> Tuple[List[Query], np.array]:
        """
        Uses the determinantal point process (DPP) based method to find a batch of queries. See
        `Batch Active Learning Using Determinantal Point Processes <https://arxiv.org/abs/1906.07975>`_ for
        more information about the method.
        
        Args:
            acquisition_func (Callable): the acquisition function to be maximized by each individual query.
            belief (Belief): the current belief distribution over the user.
            initial_query (Query): an initial query such that the output query will have the same type.
            batch_size (int): the batch size of the output.
            **kwargs: Hyperparameters `reduced_size`, `distance`, `gamma`, and extra arguments needed for specific acquisition functions.
                
                - `reduced_size` (int): The hyperparameter `B` in the original method. This method first greedily chooses `B` queries from the feasible set of queries out of the trajectory set, and then applies the boundary medoids selection. Defaults to 100.
                - `distance` (Callable): A distance function which returns a pairwise distance matrix (numpy.array) when inputted a list of queries. Defaults to :py:meth:`aprel.utils.batch_utils.default_query_distance`.
                - `gamma` (float): The hyperparameter `gamma` in the original method. The higher gamma, the more important the acquisition function values. The lower gamma, the more important the diversity of queries. Defaults to 1.

        Returns:
                2-tuple:
                
                    - List[Query]: The optimized batch of queries as a list.
                    - numpy.array: An array of floats that keep the acquisition function values corresponding to the output queries.
        """
        kwargs.setdefault('reduced_size', 100)
        kwargs.setdefault('distance', default_query_distance)
        kwargs.setdefault('gamma', 1.)
        top_queries, vals = self.greedy_batch(acquisition_func, belief, initial_query, batch_size=kwargs['reduced_size'], **kwargs)
        del kwargs['reduced_size']
        vals = vals ** kwargs['gamma']
        del kwargs['gamma']
        distances = kwargs['distance'](top_queries, **kwargs)
        ids = dpp_mode(distances, vals, batch_size)
        return [top_queries[i] for i in ids], vals[ids]