"""
A Work-In-Progress agent using Tensorforce
"""
from . import BaseAgent
from .. import characters


class TensorForceAgent(BaseAgent):
    """The TensorForceAgent. Acts through the algorith, not here."""

    def __init__(self, character=characters.Bomber, algorithm='ppo'):
        super(TensorForceAgent, self).__init__(character)
        self.algorithm = algorithm

    def act(self, obs, action_space):
        """This agent has its own way of inducing actions. See train_with_tensorforce."""
        return None

    def initialize(self, env ,numprocs):
        from gym import spaces
        from tensorforce.agents import PPOAgent

        if self.algorithm == "ppo":
            if type(env.action_space) == spaces.Tuple:
                actions = {
                    str(num): {
                        'type': int,
                        'num_values': space.n
                    }
                    for num, space in enumerate(env.action_space.spaces)
                }
            else:
                actions = dict(type='int', num_values=env.action_space.n)

            return PPOAgent(
                states=dict(type='float', shape=env.observation_space.shape),
                actions=actions,
                network=[
                    dict(type='dense', size=64),
                    dict(type='dense', size=64)
                ],
                learning_rate=1e-4,
                max_episode_timesteps=2000,
                parallel_interactions=numprocs
                )
        return None
