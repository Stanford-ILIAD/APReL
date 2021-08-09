from typing import List, Union
from copy import deepcopy
import itertools
import numpy as np

from basics import Trajectory, TrajectorySet

# TODO: OrdinalQuery classes will be implemented so that the library will include ordinal data, which was used for reward learning in:
# K. Li, M. Tucker, E. Biyik, E. Novoseller, J. W. Burdick, Y. Sui, D. Sadigh, Y. Yue, A. D. Ames;
# "ROIAL: Region of Interest Active Learning for Characterizing Exoskeleton Gait Preference Landscapes", ICRA'21.

class Query:
    """An abstract parent class that is useful for typing."""
    def __init__(self):
        pass
        
    def copy(self):
        return deepcopy(self)


class QueryWithResponse:
    """An abstract parent class that is useful for typing."""
    def __init__(self, query: Query):
        self.query = query


class DemonstrationQuery(Query):
    """A demonstration query is one where the initial state is given to the user, and they are asked to control the robot.
    
    Although not practical for optimization, this is defined for coherence."""
    def __init__(self, initial_state: np.array):
        super(DemonstrationQuery, self).__init__()
        self.initial_state = initial_state
        
        
class Demonstration(QueryWithResponse):
    def __init__(self, trajectory: Trajectory, query: DemonstrationQuery = None):
        # It is not consistent to put the query as the second argument,
        # but let's keep it because the demonstrations are only passively collected.
        initial_state, _ = trajectory[0]
        if query is None:
            query = DemonstrationQuery(initial_state)
        else:
            assert(np.all(np.isclose(query.initial_state, initial_state))), 'Mismatch between the query and the response for the demonstration.'
        super(Demonstration, self).__init__(query)
        self.trajectory = trajectory
        self.features = trajectory.features


class PreferenceQuery(Query):
    def __init__(self, slate: Union[TrajectorySet, List[Trajectory]]):
        super(PreferenceQuery, self).__init__()
        assert isinstance(slate, TrajectorySet) or isinstance(slate, list), 'Query constructor requires a TrajectorySet object for the slate.'
        self.slate = slate
        assert(self.K >= 2), 'Preference queries have to include at least 2 trajectories.'
    
    @property
    def slate(self) -> TrajectorySet:
        return self._slate
    
    @slate.setter
    def slate(self, new_slate: Union[TrajectorySet, List[Trajectory]]):
        self._slate = new_slate if isinstance(new_slate, TrajectorySet) else TrajectorySet(new_slate)
        self.K = self._slate.size
        self.response_set = np.arange(self.K)
        

class Preference(QueryWithResponse):
    def __init__(self, query: PreferenceQuery, response: int):
        super(Preference, self).__init__(query)
        assert(response in self.query.response_set), 'Response ' + str(response) + ' is out of bounds for a slate size of ' + str(self.query.K) + '.'
        self.response = response


class WeakComparisonQuery(Query):
    def __init__(self, slate: Union[TrajectorySet, List[Trajectory]]):
        super(WeakComparisonQuery, self).__init__()
        assert isinstance(slate, TrajectorySet) or isinstance(slate, list), 'Query constructor requires a TrajectorySet object for the slate.'
        self.slate = slate
        assert(self.K == 2), 'Weak comparison queries can only be pairwise comparisons, but ' + str(self.K) + ' trajectories were given.'
    
    @property
    def slate(self) -> TrajectorySet:
        return self._slate
    
    @slate.setter
    def slate(self, new_slate: Union[TrajectorySet, List[Trajectory]]):
        self._slate = new_slate if isinstance(new_slate, TrajectorySet) else TrajectorySet(new_slate)
        self.K = self._slate.size
        self.response_set = np.array([-1,0,1])


class WeakComparison(QueryWithResponse):
    def __init__(self, query: WeakComparisonQuery, response: int):
        super(WeakComparison, self).__init__(query, response)
        assert(response in self.query.response_set), 'Invalid response ' + str(response) +  ' for the weak comparison query.'
        self.response = response


class FullRankingQuery(Query):
    def __init__(self, slate: Union[TrajectorySet, List[Trajectory]]):
        super(FullRankingQuery, self).__init__()
        assert isinstance(slate, TrajectorySet) or isinstance(slate, list), 'Query constructor requires a TrajectorySet object for the slate.'
        self.slate = slate
        assert(self.K >= 2), 'Ranking queries have to include at least 2 trajectories.'
    
    @property
    def slate(self) -> TrajectorySet:
        return self._slate
    
    @slate.setter
    def slate(self, new_slate: Union[TrajectorySet, List[Trajectory]]):
        self._slate = new_slate if isinstance(new_slate, TrajectorySet) else TrajectorySet(new_slate)
        self.K = self._slate.size
        self.response_set = np.array([list(tup) for tup in itertools.permutations(np.arange(self.K))])


class FullRanking(QueryWithResponse):
    def __init__(self, query: FullRankingQuery, response: List[int]):
        super(FullRanking, self).__init__(query)
        assert(response in self.query.response_set), 'Invalid response ' + str(response) + ' for the ranking query of size ' + str(self.query.K) + '.'
        self.response = response
