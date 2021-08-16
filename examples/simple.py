import aprel
import numpy as np
import gym

env_name = 'MountainCarContinuous-v0'
gym_env = gym.make(env_name)

np.random.seed(0)
gym_env.seed(0)

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


env = aprel.Environment(gym_env, feature_func)

trajectory_set = aprel.generate_trajectories_randomly(env, num_trajectories=10,
                                                      max_episode_length=300,
                                                      file_name=env_name, seed=0)
features_dim = len(trajectory_set[0].features)

query_optimizer = aprel.QueryOptimizerDiscreteTrajectorySet(trajectory_set)

true_user = aprel.HumanUser(delay=0.5)

params = {'weights': aprel.util_funs.get_random_normalized_vector(features_dim)}
user_model = aprel.SoftmaxUser(params)
belief = aprel.SamplingBasedBelief(user_model, [], params)
print('Estimated user parameters: ' + str(belief.mean))
                                       
query = aprel.PreferenceQuery(trajectory_set[:2])

for query_no in range(10):
    queries, objective_values = query_optimizer.optimize('mutual_information', belief, query)
    print('Objective Value: ' + str(objective_values[0]))
    
    responses = true_user.respond(queries[0])
    belief.update(aprel.Preference(queries[0], responses[0]))
    print('Estimated user parameters: ' + str(belief.mean))
