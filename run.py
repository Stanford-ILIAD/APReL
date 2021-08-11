from typing import List, Tuple
import numpy as np
import gym

from basics import Environment
from utils import generate_trajectories, uniform_logprior, gaussian_proposal, util_funs, default_query_distance
from querying import QueryOptimizerDiscreteTrajectorySet
from learning import SoftmaxUser, HumanUser, TrueBelief
from learning import PreferenceQuery, Preference, FullRankingQuery, FullRanking, WeakComparisonQuery, WeakComparison
from assessing import cosine_similarity

optimization_method = 'successive_elimination'  # options: exhaustive_search, greedy, medoids, boundary_medoids, successive_elimination, dpp
batch_size = 4
acquisition_function = 'mutual_information' # options: mutual_information, volume_removal, disagreement, regret, random, thompson
max_episode_length = 100
log_prior_belief = uniform_logprior
# Method-specific parameters:
distance_metric_for_batch_generation = default_query_distance # all methods default to default_query_distance, so no need to specify
reduced_size_for_batch_generation = 100
gamma_for_dpp = 1
# Parameters for human user:
headless = False


def feature_func(traj: List[Tuple[np.array, np.array]]) -> np.array:
    """Returns the features of the given MountainCar trajectory, i.e. \Phi(traj).
    
    Args:
        traj: List of state-action tuples, e.g. [(state0, action0), (state1, action1), ...]
    
    Returns:
        features: a numpy vector corresponding the features of the trajectory
    """
    
    states = np.array([pair[0] for pair in traj])
    actions = np.array([pair[1] for pair in traj[:-1]])
    return np.random.randn(4,) # so that we don't get correlations between the features

gym_env = gym.make('MountainCarContinuous-v0')
env = Environment(gym_env, feature_func)


np.random.seed(0)
trajectory_set = generate_trajectories(env, num_trajectories=40, max_episode_length=max_episode_length,
                                       file_name='mountaincar', restore=False, headless=headless)
                                       
# Initialize a dummy query. The optimizer will then find the optimal query of the same kind.
query = WeakComparisonQuery(trajectory_set[:2])

query_optimizer = QueryOptimizerDiscreteTrajectorySet(trajectory_set)



features_dim = len(trajectory_set[0].features)

#true_params = {'omega': util_funs.get_random_normalized_vector(features_dim)}
#true_user = SoftmaxUser(true_params)
true_user = HumanUser()

current_params = {'omega': util_funs.get_random_normalized_vector(features_dim)}
user_model = SoftmaxUser(current_params)

dataset = []

belief = TrueBelief(log_prior_belief, user_model, dataset, current_params, num_samples = 100,
                    proposal_distribution = gaussian_proposal, burnin = 200, thin = 20)
                    
for query_no in range(10):
    print('Estimated user parameters: ' + str(belief.mean))
    if not isinstance(true_user, HumanUser):
        cos_sim = cosine_similarity(belief, true_user)
        print('Cosine Similarity: ' + str(cos_sim))
    current_params = belief.mean

    queries, objective_values = query_optimizer.optimize(acquisition_function, belief,
                                                         query, batch_size=batch_size, 
                                                         optimization_method=optimization_method,
                                                         reduced_size=reduced_size_for_batch_generation,
                                                         gamma=gamma_for_dpp,
                                                         distance=default_query_distance)
    print('Objective Values: ' + str(objective_values))
    responses = true_user.respond(queries)
    
    belief.update([Preference(query, response) for query, response in zip(queries, responses)])

belief = TrueBelief(log_prior_belief, user_model, dataset, current_params,
                    proposal_distribution = gaussian_proposal, num_samples = 100,
                    burnin = 200, thin = 20)
print('Estimated user parameters: ' + str(belief.mean))
if not isinstance(true_user, HumanUser):
    cos_sim = cosine_similarity(belief, true_user)
    print('Cosine Similarity: ' + str(cos_sim))