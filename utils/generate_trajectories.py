from typing import List, Union
import pickle
import numpy as np
from moviepy.editor import ImageSequenceClip
import warnings
import os

from basics import Environment, Trajectory, TrajectorySet

def generate_trajectories(env: Environment,
                          num_trajectories: int,
                          max_episode_length: int = None,
                          file_name: str = None,
                          restore: bool = False,
                          headless: bool = False) -> TrajectorySet:
    """
    Generates num_trajectories random trajectories, or loads them from the given file.
    Args:
        env: an Environment, which is a class containing an OpenAI Gym environment and a features function
        num_trajectories: the number of trajectories to generate
        max_episode_length: the maximum number of time steps for the new trajectories
        file_name: the file name
        restore: if true, will first try to load the trajectories from file_name
        headless: if true, trajectories are saved and printed without rendering

    Returns: a TrajectorySet of num_trajectories trajectories

    """
    assert(not (file_name is None and restore)), 'Trajectory set cannot be restored, because no file_name is given.'
    max_episode_length = np.inf if max_episode_length is None else max_episode_length
    if restore:
        try:
            with open('data/' + file_name + '.pkl', 'rb') as f:
                trajectories = pickle.load(f)
        except:
            warnings.warn('Ignoring restore=True, because \'data/' + file_name + '.pkl\' is not found.')
            trajectories = TrajectorySet([])
        if not headless:
            for traj_no in range(trajectories.size):
                if not os.path.isfile(trajectories[traj_no].clip_path):
                    warnings.warn('Ignoring restore=True, because headless=False and some trajectory clips are missing.')
                    trajectories = TrajectorySet([])
                    break
    else:
        trajectories = TrajectorySet([])
    
    if not os.path.exists('data'):
        os.makedirs('data')
    
    if trajectories.size >= num_trajectories:
        trajectories = TrajectorySet(trajectories[:num_trajectories])
    else:
        env_has_rgb_render = env.render_exists and not headless
        if env_has_rgb_render and not os.path.exists('data/clips'):
            os.makedirs('data/clips')
        for traj_no in range(trajectories.size, num_trajectories):
            traj = []
            obs = env.reset()
            if env_has_rgb_render:
                try:
                    frames = [np.uint8(env.render(mode='rgb_array'))]
                except:
                    env_has_rgb_render = False
            done = False
            t = 0
            while not done and t < max_episode_length:
                act = env.action_space.sample()
                traj.append((obs,act))
                obs, _, done, _ = env.step(act)
                t += 1
                if env_has_rgb_render:
                    frames.append(np.uint8(env.render(mode='rgb_array')))
            traj.append((obs, None))
            if env_has_rgb_render:
                clip = ImageSequenceClip(frames, fps=25)
                clip_path = 'data/clips/' + file_name + str(traj_no) + '.mp4'
                clip.write_videofile(clip_path, audio=False)
            else:
                clip_path = None
            trajectories.append(Trajectory(env, traj, clip_path))
            if env.close_exists:
                env.close()

    with open('data/' + file_name + '.pkl', 'wb') as f:
        pickle.dump(trajectories, f)

    if not headless and trajectories[-1].clip_path is None:
        warnings.warn(('headless=False was set, but either the environment is missing '
                       'a render function or the render function does not accept '
                       'mode=\'rgb_array\'. Automatically switching to headless mode.'))
    if headless:
        for traj_no in range(trajectories.size):
            trajectories[traj_no].clip_path = None
    return trajectories
