from typing import List, Tuple, Union
import numpy as np

from basics import Environment


class Trajectory:
    def __init__(self, env: Environment, trajectory: List[Tuple[np.array, np.array]]):
        self.env = env
        self.trajectory = trajectory
        self.features = env.features(trajectory)
        
    def __getitem__(self, t: int) -> Tuple[np.array, np.array]:
        return self.trajectory[t]


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