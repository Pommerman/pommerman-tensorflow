"""Train an agent with TensorForce.

Call this with a config, a game, and a list of agents, one of which should be a
tensorforce agent. The script will start separate threads to operate the agents
and then report back the result.

An example with all three simple agents running ffa:
python train_with_tensorforce.py --agents tensorforce::ppo, test::agents.SimpleAgent, test::agents.SimpleAgent, test::agents.SimpleAgent --config=PommeFFACompetition-v0

python train_with_tensorforce.py --agents "tensorforce::ppo,random::,random::,random::" --config=PommeFFACompetition-v0 --episodes 100 --batch-size 1 --modelname testing 

python train_with_tensorforce.py --agents "tensorforce::ppo,test::agents.SimpleAgent,test::agents.SimpleAgent,test::agents.SimpleAgent" --config=PommeFFACompetition-v0 --episodes 1000 --batch-size 100 --modelname rewardsv0

"""
import atexit
import functools
import os

import argparse
import docker
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym
import gym

from pommerman import helpers, make
from pommerman.agents import TensorForceAgent

import timeit


import matplotlib.pyplot as plt
import pickle
import numpy as np
CLIENT = docker.from_env()


def clean_up_agents(agents):
    """Stops all agents"""
    return [agent.shutdown() for agent in agents]


class WrappedEnv(OpenAIGym):
    '''An Env Wrapper used to make it easier to work
    with multiple agents'''

    def __init__(self, gym, visualize=False):
        self.gym = gym
        self.visualize = visualize

    def execute(self, action):
        if self.visualize:
            self.gym.render()

        actions = self.unflatten_action(action=action)

        obs = self.gym.get_observations()
        all_actions = self.gym.act(obs)
        all_actions.insert(self.gym.training_agent, actions)
        state, reward, terminal, _ = self.gym.step(all_actions)
        agent_state = self.gym.featurize(state[self.gym.training_agent])
        agent_reward = reward[self.gym.training_agent]
        return agent_state, terminal, agent_reward

    def reset(self):
        obs = self.gym.reset()
        agent_obs = self.gym.featurize(obs[3])
        return agent_obs


def main():
    '''CLI interface to bootstrap taining'''
    parser = argparse.ArgumentParser(description="Playground Flags.")
    parser.add_argument("--game", default="pommerman", help="Game to choose.")
    parser.add_argument(
        "--config",
        default='PommeFFACompetition-v0',
        help="Configuration to execute. See env_ids in "
        "configs.py for options. default is 1v1")
    parser.add_argument(
        "--agents",
        default="tensorforce::ppo,test::agents.SimpleAgent,"
        "test::agents.SimpleAgent,test::agents.SimpleAgent",
        help="Comma delineated list of agent types and docker "
        "locations to run the agents.")
        #agent in position 1

    parser.add_argument(
        "--agent_env_vars",
        help="Comma delineated list of agent environment vars "
        "to pass to Docker. This is only for the Docker Agent."
        " An example is '0:foo=bar:baz=lar,3:foo=lam', which "
        "would send two arguments to Docker Agent 0 and one to"
        " Docker Agent 3.",
        default="")
    parser.add_argument(
        "--record_pngs_dir",
        default=None,
        help="Directory to record the PNGs of the game. "
        "Doesn't record if None.")
    parser.add_argument(
        "--record_json_dir",
        default=None,
        help="Directory to record the JSON representations of "
        "the game. Doesn't record if None.")
    parser.add_argument(
        "--render",
        default=False,
        action='store_true',
        help="Whether to render or not. Defaults to False.")
    parser.add_argument(
        "--game_state_file",
        default=None,
        help="File from which to load game state. Defaults to "
        "None.")
    parser.add_argument(
            '--batch-size',
            default=100,
            type=int,
            help='average reward visualization by batch size. default=100 episodes'
            )
    parser.add_argument(
            '--episodes',
            default=1000,
            type=int,
            help='number of training episodes, default=1000. must be divisible by batch_size'
            )
    parser.add_argument(
            '--modelname',
            default='default',
            help='name of model file, default= default.ckpt'
            )
    parser.add_argument(
            '--loadfile',
            default=False,
            action='store_true',
            help='sets true to load prev model'
            )
    args = parser.parse_args()

    config = args.config
    record_pngs_dir = args.record_pngs_dir
    record_json_dir = args.record_json_dir
    agent_env_vars = args.agent_env_vars
    game_state_file = args.game_state_file


    #variables    
    save_path='saved_models\\'
    model_name=args.modelname+'.ckpt'
    batch_size=args.batch_size
    num_episodes=args.episodes
    assert(num_episodes%batch_size==0)
    # TODO: After https://github.com/MultiAgentLearning/playground/pull/40
    #       this is still missing the docker_env_dict parsing for the agents.
    agents = [
        helpers.make_agent_from_string(agent_string, agent_id + 1000)
        for agent_id, agent_string in enumerate(args.agents.split(","))
    ]

    env = make(config, agents, game_state_file)
    training_agent = None

    for agent in agents:
        if type(agent) == TensorForceAgent:
            training_agent = agent
            env.set_training_agent(agent.agent_id)
            break

    if args.record_pngs_dir:
        assert not os.path.isdir(args.record_pngs_dir)
        os.makedirs(args.record_pngs_dir)
    if args.record_json_dir:
        assert not os.path.isdir(args.record_json_dir)
        os.makedirs(args.record_json_dir)

    # Create a Proximal Policy Optimization agent
    agent = training_agent.initialize(env)

    if args.loadfile:
        agent.restore_model(save_path)
    
    atexit.register(functools.partial(clean_up_agents, agents))
    wrapped_env = WrappedEnv(env, visualize=args.render)
    
    runner_time = timeit.default_timer()
    
    #load history.pickle
    if args.loadfile:
        try :
            handle = open(save_path+'history.pickle','rb')
            history=pickle.load(handle)
        except:
            history=None
    else:
        history=None
    if history:
        runner = Runner(agent=agent, environment=wrapped_env,history=history)
    else:
        runner = Runner(agent=agent, environment=wrapped_env)
    runner.run(episodes=num_episodes, max_episode_timesteps=2000)
   #print("Stats: ", runner.episode_rewards, runner.episode_timesteps,
        #runner.episode_times)
    print(runner.episode_rewards)
    
    history={}
    history['episode_rewards']=runner.episode_rewards
    history['episode_timesteps']=runner.episode_timesteps
    history['episode_times']=runner.episode_times
    with open(save_path+'history.pickle','wb') as handle:
        pickle.dump(history,handle)
    agent.save_model(save_path+model_name, False)
    print('Runner time: ', timeit.default_timer() - runner_time)

    plt.plot(np.arange(0,int(len(runner.episode_rewards)/batch_size)),np.mean(np.asarray(runner.episode_rewards).reshape(-1,batch_size),axis=1))
    plt.title('average rewards per batch of episodes')
    plt.ylabel('average reward')
    plt.xlabel('batch of ' +str(batch_size)+' episodes')
    plt.show()    
    
    try:
        runner.close()
    except AttributeError as e:
        pass


if __name__ == "__main__":
    main()
