import aprel
import numpy as np
import gym


def feature_func(traj):
    """Returns the features of the given MountainCar trajectory, i.e. \Phi(traj).
    
    Args:
        traj: List of state-action tuples, e.g. [(state0, action0), (state1, action1), ...]
    
    Returns:
        features: a numpy vector corresponding the features of the trajectory
    """
    states = np.array([pair[0] for pair in traj])
    actions = np.array([pair[1] for pair in traj[:-1]])
    min_pos, max_pos = states[:,0].min(), states[:,0].max()
    mean_speed = np.abs(states[:,1]).mean()
    mean_vec = [-0.703, -0.344, 0.007]
    std_vec = [0.075, 0.074, 0.003]
    return (np.array([min_pos, max_pos, mean_speed]) - mean_vec) / std_vec


def main(args):
    # Create the OpenAI Gym environment
    gym_env = gym.make(args['env'])
    
    # Seed for reproducibility
    np.random.seed(args['seed'])
    gym_env.seed(args['seed'])

    # Wrap the environment with a feature function
    env = aprel.Environment(gym_env, args['feature_func'])

    # Create a trajectory set
    trajectory_set = aprel.generate_trajectories_randomly(env, num_trajectories=args['num_trajectories'],
                                                          max_episode_length=args['max_episode_length'],
                                                          file_name=args['env'], restore=args['restore'],
                                                          headless=args['headless'], seed=args['seed'])
    features_dim = len(trajectory_set[0].features)

    # Initialize the query optimizer
    query_optimizer = aprel.QueryOptimizerDiscreteTrajectorySet(trajectory_set)

    # Initialize the object for the true human
    if args['simulate']:
        true_params = {'weights': aprel.util_funs.get_random_normalized_vector(features_dim)}
        true_user = aprel.SoftmaxUser(true_params)
    else:
        true_user = aprel.HumanUser(delay=args['human_visualization_delay'])
    
    # Create the human response model and initialize the belief distribution
    params = {'weights': aprel.util_funs.get_random_normalized_vector(features_dim)}
    user_model = aprel.SoftmaxUser(params)
    belief = aprel.SamplingBasedBelief(user_model, [], params, logprior=args['log_prior_belief'],
                                       num_samples=args['num_samples'],
                                       proposal_distribution=args['proposal_distribution'],
                                       burnin=args['burnin'], thin=args['thin'])
    # Report the metrics
    print('Estimated user parameters: ' + str(belief.mean))
    if args['simulate']:
        cos_sim = aprel.cosine_similarity(belief, true_user)
        print('Cosine Similarity: ' + str(cos_sim))

    # Initialize a dummy query so that the query optimizer will generate queries of the same kind
    if args['query_type'] == 'preference':
        query = aprel.PreferenceQuery(trajectory_set[:args['query_size']])
    elif args['query_type'] == 'weak_comparison':
        query = aprel.WeakComparisonQuery(trajectory_set[:args['query_size']])
    elif args['query_type'] == 'full_ranking':
        query = aprel.FullRankingQuery(trajectory_set[:args['query_size']])
    else:
        raise NotImplementedError('Unknown query type.')

    # Active learning loop
    for query_no in range(args['num_iterations']):
        # Optimize the query
        queries, objective_values = query_optimizer.optimize(args['acquisition'], belief,
                                                             query, batch_size=args['batch_size'], 
                                                             optimization_method=args['optim_method'],
                                                             reduced_size=args['reduced_size_for_batches'],
                                                             gamma=args['dpp_gamma'],
                                                             distance=args['distance_metric_for_batches'])
        print('Objective Values: ' + str(objective_values))

        # Ask the query to the human
        responses = true_user.respond(queries)
        
        #Update the belief distribution
        belief.update([aprel.Preference(query, response) for query, response in zip(queries, responses)])
        
        # Report the metrics
        print('Estimated user parameters: ' + str(belief.mean))
        if args['simulate']:
            cos_sim = aprel.cosine_similarity(belief, true_user)
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
    parser.add_argument('--max_episode_length', type=int, default=None,
                        help='Maximum number of time steps per episode ONLY FOR the new trajectories. Defaults to no limit.')
    parser.add_argument('--restore', dest='restore', action='store_true',
                        help='Use this flag if you want to restore the discrete trajectory set from the existing data folder.')
    parser.set_defaults(restore=False)
    parser.add_argument('--headless', dest='headless', action='store_true',
                        help='Use this flag if you want to run the code in a headless way, i.e., with no visualization.')
    parser.set_defaults(headless=False)
    parser.add_argument('--simulate', dest='simulate', action='store_true',
                        help='Use this flag if you want to run the code with simulated synthetic users who follow a softmax model.')
    parser.set_defaults(simulate=False)
    parser.add_argument('--human_visualization_delay', type=float, default=0.5,
                        help='Delay between each trajectory visualization during querying.')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples for the sampling based belief.')
    parser.add_argument('--burnin', type=int, default=200,
                        help='Number of burn-in steps for Metropolis-Hastings in the sampling based belief.')
    parser.add_argument('--thin', type=int, default=20,
                        help='Thinning parameter for Metropolis-Hastings in the sampling based belief.')
    parser.add_argument('--query_type', type=str, default='preference',
                        help='Type of the queries that will be actively asked to the user. Options: preference, weak_comparison, full_ranking.')
    parser.add_argument('--query_size', type=int, default=2,
                        help='Number of trajectories in each query.')
    parser.add_argument('--num_iterations', type=int, default=10,
                        help='Number of iterations in the active learning loop.')
    parser.add_argument('--optim_method', type=str, default='exhaustive_search',
                        help='Options: exhaustive_search, greedy, medoids, boundary_medoids, successive_elimination, dpp.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size can be set >1 for batch active learning algorithms.')
    parser.add_argument('--acquisition', type=str, default='random',
                        help='Acquisition function for active querying. Options: mutual_information, volume_removal, disagreement, regret, random, thompson')
    parser.add_argument('--reduced_size_for_batches', type=int, default=100,
                        help='The number of greedily chosen candidate queries (reduced set) for batch generation.')
    parser.add_argument('--dpp_gamma', type=int, default=1,
                        help='Gamma parameter for the DPP method: the higher gamma the more important is the acquisition function relative to diversity.')

    args = vars(parser.parse_args())
    args['feature_func'] = feature_func
    args['log_prior_belief'] = aprel.uniform_logprior
    args['proposal_distribution'] = aprel.gaussian_proposal
    args['distance_metric_for_batches'] = aprel.default_query_distance # all relevant methods default to default_query_distance

    main(args)
