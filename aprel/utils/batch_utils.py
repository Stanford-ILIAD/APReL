"""Utility functions for active batch generation."""

from typing import List
import numpy as np
import scipy.spatial.distance as ssd

from aprel.learning import Query, PreferenceQuery, WeakComparisonQuery, FullRankingQuery


def default_query_distance(queries: List[Query], **kwargs) -> np.array:
    """Given a set of m queries, returns an m-by-m matrix, each entry representing the distance between the corresponding queries.
    
    Args:
        queries (List[Query]): list of m queries for which the distances will be computed
        **kwargs: The hyperparameters.

            - `metric` (str): The distance metric can be specified with this argument. Defaults to 'euclidean'. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html for the set of available metrics.
                  
    Returns:
        numpy.array: an m-by-m numpy array that consists of the pairwise distances between the queries.
        
    Raises:
        AssertionError: if the query is not a compatible type. Currently, the compatible types are: :class:`.FullRankingQuery`, :class:`.PreferenceQuery`, and :class:`.WeakComparisonQuery` (all for a slate size of 2).
    """
    kwargs.setdefault('metric', 'euclidean')
    compatible_types = [isinstance(query, PreferenceQuery) or isinstance(query, WeakComparisonQuery) or isinstance(query, FullRankingQuery) for query in queries]
    assert np.all(compatible_types), 'Default query distance, which you are using for batch selection, does not support the given query types. Consider using a custom distance function. See utils/batch_utils.py.'
    assert np.all([query.K == 2 for query in queries]), 'Default query distance, which you are using for batch selection, does not support large slates, use K = 2. Or consider using a custom distance function. See utils/batch_utils.py.'

    features_diff = [query.slate.features_matrix[0] - query.slate.features_matrix[1] for query in queries]
    return ssd.squareform(ssd.pdist(features_diff, kwargs['metric']))
