from typing import Callable, List, Tuple
import itertools
import numpy as np
from scipy.spatial import ConvexHull
import warnings

from basics import Trajectory, TrajectorySet
from learning import Belief, TrueBelief
from learning import Query, PreferenceQuery, WeakComparisonQuery, FullRankingQuery
from querying import mutual_information, volume_removal, disagreement, regret, random, thompson
from utils import kMedoids, dpp_mode, default_query_distance


class QueryOptimizer:
    def __init__(self):
        self.acquisition_functions = {'mutual_information': mutual_information,
                                      'volume_removal': volume_removal,
                                      'disagreement': disagreement,
                                      'regret': regret,
                                      'random': random,
                                      'thompson': thompson}


class QueryOptimizerDiscreteTrajectorySet(QueryOptimizer):
    def __init__(self, trajectory_set: TrajectorySet):
        super(QueryOptimizerDiscreteTrajectorySet, self).__init__()
        self.trajectory_set = trajectory_set
        
    def argplanner(self, omega: np.array) -> int:
        return np.asscalar(np.argmax(np.dot(self.trajectory_set.features_matrix, omega)))

    def planner(self, omega: np.array) -> Trajectory:
        return self.trajectory_set[self.argplanner(omega)]
        
    def optimize(self,
                 acquisition_func_str: str,
                 belief: Belief,
                 initial_query: Query,
                 batch_size: int = 1,
                 optimization_method: str = 'exhaustive_search',
                 **kwargs) -> Tuple[List[Query], np.array]:
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
        return self.greedy_batch(acquisition_func, belief, initial_query, batch_size=1, **kwargs)

    def greedy_batch(self,
                     acquisition_func: Callable,
                     belief: Belief,
                     initial_query: Query,
                     batch_size: int,
                     **kwargs) -> Tuple[List[Query], np.array]:
        if isinstance(initial_query, PreferenceQuery) or isinstance(initial_query, WeakComparisonQuery) or isinstance(initial_query, FullRankingQuery):
            if acquisition_func is random:
                best_batch = [initial_query.copy() for _ in range(batch_size)]
                for i in range(batch_size):
                    best_batch[i].slate = self.trajectory_set[np.random.choice(self.trajectory_set.size, size=initial_query.K, replace=False)]
                return best_batch, np.array([1. for _ in range(batch_size)])
                
            elif acquisition_func is thompson and isinstance(belief, TrueBelief):
                subsets = np.array([list(tup) for tup in itertools.combinations(np.arange(belief.num_samples), initial_query.K)])
                planned_traj_ids = [self.argplanner(sample['omega']) for sample in belief.samples]
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
                
            elif acquisition_func is disagreement and isinstance(belief, TrueBelief):
                assert(initial_query.K == 2), 'disagreement acquisition function works only with pairwise comparison queries, i.e., K must be 2.'
                subsets = np.array([list(tup) for tup in itertools.combinations(np.arange(belief.num_samples), initial_query.K)])
                vals = []
                belief_samples = np.array(belief.samples)
                belief_logprobs = np.array(belief.logprobs)
                for ids in subsets:
                    omegas = np.array([sample['omega'] for sample in belief_samples[ids]])
                    vals.append(acquisition_func(omegas, belief_logprobs[ids], **kwargs))
                vals = np.array(vals)
                inds = np.argpartition(vals, -batch_size)[-batch_size:]

                best_batch = [initial_query.copy() for _ in range(batch_size)]
                for i in range(batch_size):
                    best_batch[i].slate = TrajectorySet([self.planner(belief.samples[best_id]['omega']) for best_id in subsets[inds[i]]])
                return best_batch, vals[inds]

                best_query = initial_query.copy()
                best_query.slate = [self.planner(belief.samples[best_id]['omega']) for best_id in best_ids]
                return best_query, maxval
                
            elif acquisition_func is regret and isinstance(belief, TrueBelief):
                assert(initial_query.K == 2), 'regret acquisition function works only with pairwise comparison queries, i.e., K must be 2.'
                subsets = np.array([list(tup) for tup in itertools.combinations(np.arange(belief.num_samples), initial_query.K)])
                planned_trajs = TrajectorySet([self.planner(sample['omega']) for sample in belief.samples])
                vals = []
                belief_samples = np.array(belief.samples)
                belief_logprobs = np.array(belief.logprobs)
                for ids in subsets:
                    omegas = np.array([sample['omega'] for sample in belief_samples[ids]])
                    vals.append(acquisition_func(omegas, belief_logprobs[ids], planned_trajs[ids], **kwargs))
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
        kwargs.setdefault('reduced_size', 100)
        kwargs.setdefault('distance', default_query_distance)
        top_queries, vals = self.greedy_batch(acquisition_func, belief, initial_query, batch_size=kwargs['reduced_size'], **kwargs)
        del kwargs['reduced_size']
        distances = kwargs['distance'](top_queries, **kwargs)
        medoid_ids, _ = kMedoids(distances, batch_size)
        return [top_queries[idx] for idx in medoid_ids], vals[medoid_ids]
        
    def boundary_medoids_batch(self,
                               acquisition_func: Callable,
                               belief: Belief,
                               initial_query: Query,
                               batch_size: int,
                               **kwargs) -> Tuple[List[Query], np.array]:
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
            medoid_ids = np.concatenate(medoid_ids, ids_to_add)
        else:
            # Otherwise, select the medoids among the boundary queries
            distances = kwargs['distance']([top_queries[i] for i in simplices], **kwargs)
            temp_ids, _ = kMedoids(distances, batch_size)
            medoid_ids = simplices[temp_ids]
        return [top_queries[idx] for idx in medoid_ids], vals[medoid_ids]

    def successive_elimination_batch(self,
                                     acquisition_func: Callable,
                                     belief: Belief,
                                     initial_query: Query,
                                     batch_size: int,
                                     **kwargs) -> Tuple[List[Query], np.array]:
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