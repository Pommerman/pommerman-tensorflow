'''This is the base abstraction for agents in pommerman.
All agents should inherent from this class'''
from .. import characters


class BaseAgent:
    """Parent abstract Agent."""

    def __init__(self, character=characters.Bomber):
        self._character = character

        self.visited = []
        self.num_cells_visited = 0

    def __getattr__(self, attr):
        return getattr(self._character, attr)

    def act(self, obs, action_space):
        raise NotImplementedError()

    def episode_end(self, reward):
        """This is called at the end of the episode to let the agent know that
        the episode has ended and what is the reward.

        Args:
          reward: The single reward scalar to this agent.
        """
        pass

    def init_agent(self, id_, game_type):
        self._character = self._character(id_, game_type)

    @staticmethod
    def has_user_input():
        return False

    def shutdown(self):
        pass

    def _visit(self, cell):
        self.visited.append(cell)
        self.num_cells_visited += 1

    def getPositionReward(self, cell):
        if(cell not in self.visited):
            self._visit(cell)
            return 0.5
        else:
            self.num_cells_visited += 1
            return 0.5 / self.num_cells_visited