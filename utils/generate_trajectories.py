from typing import List
import pickle

from basics import Environment, Trajectory, TrajectorySet

def generate_trajectories(env: Environment,
                          num_trajectories: int,
                          file_name: str = None,
                          save: bool = True,
                          restore: bool = False) -> TrajectorySet:
    assert(not (file_name is None and restore)), 'Trajectory set cannot be restored, because no file_name is given.'
    assert(not (file_name is None and save)), 'Trajectory set cannot be saved, because no file_name is given.'
    if restore:
        with open(file_name, 'rb') as f:
            trajectories = pickle.load(f)
    else:
        trajectories = TrajectorySet([])
    
    if trajectories.size >= num_trajectories:
        trajectories = trajectories[:num_trajectories]
    else:
        for _ in range(trajectories.size, num_trajectories):
            traj = []
            obs = env.reset()
            done = False
            while not done:
                act = env.action_space.sample()
                traj.append((obs,act))
                obs, _, done, _ = env.step(act)
            traj.append((obs, env.action_space.sample()))
            trajectories.append(Trajectory(env, traj))
    
    if save:
        with open(file_name, 'wb') as f:
            pickle.dump(trajectories, f)

    return trajectories
