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
        self.tf_agent = None
        self.agent_id = 0
        self.env = None

    def act(self, obs, action_space):
        ppo_state = self.envWrapper(obs)
        return self.tf_agent.act(ppo_state)

    def initialize(self, env, summarizer=None):
        from gym import spaces
        from tensorforce.agents import PPOAgent

        self.env = env

        if self.algorithm == "ppo":
            if type(env.action_space) == spaces.Tuple:
                actions = {
                    str(num): {
                        'type': int,
                        'num_actions': space.n
                    }
                    for num, space in enumerate(env.action_space.spaces)
                }
            else:
                actions = dict(type='int', num_actions=env.action_space.n)

            self.tf_agent = PPOAgent(
                states=dict(type='float', shape=env.observation_space.shape),
                actions=actions,
                network=[
                    dict(type='dense', size=64),
                    dict(type='dense', size=64)
                ],
                summarizer=summarizer,
                execution={'num_parallel':64, 'type': 'single', 'session_config':None, 'distributed_spec':None},
                batching_capacity=1000,
                step_optimizer=dict(type='adam', learning_rate=1e-4))

            return self.tf_agent
        return None

    def set_agent_id(self, agent_id):
        self.agent_id = agent_id

    def restore_model(self, model_name):
        self.tf_agent.restore_model(model_name)

    def envWrapper(self, state):
        return self.env.featurize(state)
        