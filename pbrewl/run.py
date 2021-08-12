from typing import List, Tuple
import numpy as np
import gym

from basics import Environment
from utils import generate_trajectories, uniform_logprior, gaussian_proposal, util_funs, default_query_distance
from querying import QueryOptimizerDiscreteTrajectorySet
from learning import SoftmaxUser, HumanUser, SamplingBasedBelief
from learning import PreferenceQuery, Preference, FullRankingQuery, FullRanking, WeakComparisonQuery, WeakComparison
from assessing import cosine_similarity


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


def main(args):
    gym_env = gym.make(args['env'])
    env = Environment(gym_env, args['feature_func'])

    np.random.seed(args['seed'])
    trajectory_set = generate_trajectories(env, num_trajectories=args['num_trajectories'],
                                           max_episode_length=args['max_episode_length'],
                                           file_name=args['env'], restore=args['restore'], headless=args['headless'])
    features_dim = len(trajectory_set[0].features)
                                           
    # Initialize a dummy query. The optimizer will then find the optimal query of the same kind.
    if args['query_type'] == 'preference':
        query = PreferenceQuery(trajectory_set[:args['query_size']])
    elif args['query_type'] == 'weak_comparison':
        query = WeakComparisonQuery(trajectory_set[:args['query_size']])
    elif args['query_type'] == 'full_ranking':
        query = FullRankingQuery(trajectory_set[:args['query_size']])
    else:
        raise NotImplementedError('Unknown query type.')

    query_optimizer = QueryOptimizerDiscreteTrajectorySet(trajectory_set)

    if args['simulate']:
        true_params = {'omega': util_funs.get_random_normalized_vector(features_dim)}
        true_user = SoftmaxUser(true_params)
    else:
        true_user = HumanUser()

    current_params = {'omega': util_funs.get_random_normalized_vector(features_dim)}
    user_model = SoftmaxUser(current_params)

    belief = SamplingBasedBelief(args['log_prior_belief'], user_model, [], current_params,
                        num_samples=100, proposal_distribution=gaussian_proposal, burnin=200, thin=20)
    print('Estimated user parameters: ' + str(belief.mean))
    if args['simulate']:
        cos_sim = cosine_similarity(belief, true_user)
        print('Cosine Similarity: ' + str(cos_sim))
    
    for query_no in range(10):
        queries, objective_values = query_optimizer.optimize(args['acquisition'], belief,
                                                             query, batch_size=args['batch_size'], 
                                                             optimization_method=args['optim_method'],
                                                             reduced_size=args['reduced_size_for_batches'],
                                                             gamma=args['dpp_gamma'],
                                                             distance=args['distance_metric_for_batches'])
        print('Objective Values: ' + str(objective_values))
        responses = true_user.respond(queries)
        
        belief.update([Preference(query, response) for query, response in zip(queries, responses)])
        print('Estimated user parameters: ' + str(belief.mean))
        if args['simulate']:
            cos_sim = cosine_similarity(belief, true_user)
            print('Cosine Similarity: ' + str(cos_sim))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True,
                        help='The name of the OpenAI Gym environment.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for numpy randomness.')
    parser.add_argument('--num_trajectories', type=int, default=40,
                        help='Number of trajectories in the discrete trajectory set for query optimization.')
    parser.add_argument('--restore', dest='restore', action='store_true',
                        help='Use this flag if you want to restore the discrete trajectory set from the existing data folder.')
    parser.add_argument('--no-restore', dest='restore', action='store_false',
                        help='You may use this flag to create the discrete trajectory set from scratch. This is the default setting.')
    parser.set_defaults(restore=False)
    parser.add_argument('--simulate', dest='simulate', action='store_true',
                        help='Use this flag if you want to run the code with simulated synthetic users who follow a softmax model.')
    parser.add_argument('--no-simulate', dest='simulate', action='store_false',
                        help='You may use this flag to run the code with a real user who will need to respond to the queries. This is the default setting.')
    parser.set_defaults(simulate=False)
    parser.add_argument('--headless', dest='headless', action='store_true',
                        help='Use this flag if you want to run the code in a headless way, i.e., with no visualization.')
    parser.add_argument('--no-headless', dest='headless', action='store_false',
                        help='You may use this flag to run the code with visualization. This is the default setting.')
    parser.set_defaults(headless=False)
    parser.add_argument('--query_type', type=str, default='preference',
                        help='Type of the queries that will be actively asked to the user. Options: preference, weak_comparison, full_ranking.')
    parser.add_argument('--query_size', type=int, default=2,
                        help='Number of trajectories in each query.')
    parser.add_argument('--optim_method', type=str, default='exhaustive_search',
                        help='Options: exhaustive_search, greedy, medoids, boundary_medoids, successive_elimination, dpp.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size can be set >1 for batch active learning algorithms.')
    parser.add_argument('--acquisition', type=str, default='random',
                        help='Acquisition function for active querying. Options: mutual_information, volume_removal, disagreement, regret, random, thompson')
    parser.add_argument('--max_episode_length', type=int, default=None,
                        help='Maximum number of time steps per episode ONLY FOR the new trajectories. Defaults to no limit.')
    parser.add_argument('--reduced_size_for_batches', type=int, default=100,
                        help='The number of greedily chosen candidate queries (reduced set) for batch generation.')
    parser.add_argument('--dpp_gamma', type=int, default=1,
                        help='Gamma parameter for the DPP method: the higher gamma the more important is the acquisition function relative to diversity.')

    args = vars(parser.parse_args())
    args['feature_func'] = feature_func
    args['log_prior_belief'] = uniform_logprior
    args['distance_metric_for_batches'] = default_query_distance # all methods default to default_query_distance, so no need to specify

    main(args)
