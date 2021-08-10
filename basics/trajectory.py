from typing import List, Tuple, Union
import numpy as np
import time

from basics import Environment

# TODO: Currently, trajectory visualization relies on the existence of a set_state function in the environment, which is often not available.
# This also causes actions to be not visualized. Instead, the simulator moves from state to state and renders in between.
# Ideally, each trajectory should keep a visualization of the trajectory, e.g., a video. But it is computationally too costy to generate videos.

class Trajectory:
    def __init__(self, env: Environment, trajectory: List[Tuple[np.array, np.array]]):
        self.env = env
        self.trajectory = trajectory
        self.features = env.features(trajectory)
        
    def __getitem__(self, t: int) -> Tuple[np.array, np.array]:
        return self.trajectory[t]
        
    @property
    def length(self):
        return len(self.trajectory)
        
    def visualize(self, pause: float=0.0):
        if self.env.set_state is not None and self.env.render is not None:
            for t in range(self.length):
                self.env.set_state(self.trajectory[t][0])
                self.env.render()
                time.sleep(pause)
        else:
            print('Either set_state or render function is missing from the environment. Printing the trajectory instead.')
            #print(self.trajectory)
            print('Features for this trajectory are: ' + str(self.features))


class TrajectorySet:
    def __init__(self, trajectories: List[Trajectory]):
        self.trajectories = trajectories
        self.features_matrix = np.array([trajectory.features for trajectory in self.trajectories])

    def __getitem__(self, idx: Union[int, List[int], np.array]):
        if isinstance(idx, list) or type(idx).__module__ == np.__name__:
            return TrajectorySet([self.trajectories[i] for i in idx])
        return self.trajectories[idx]
        
    def __setitem__(self, idx, new_trajectory):
        self.trajectories[idx] = new_trajectory

    @property
    def size(self) -> int:
        return len(self.trajectories)
        
    def append(self, new_trajectory):
        self.trajectories.append(new_trajectory)
        if self.size == 1:
            self.features_matrix = new_trajectory.features.reshape((1,-1))
        else:
            self.features_matrix = np.vstack((self.features_matrix, new_trajectory.features))