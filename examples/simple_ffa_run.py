'''An example to show how to set up an pommerman game programmatically'''
import pommerman
from pommerman import agents

from data_collection_helper import save_game, load_game

def envWrapper(gym, state, agent_id):
    agent_state = gym.featurize(state[agent_id])
    return agent_state


def main():
    '''Simple function to bootstrap a game.
       
       Use this as an example to set up your training env.
    '''
    # Print all possible environments in the Pommerman registry
    print(pommerman.REGISTRY)

    test_agent = agents.TensorForceAgent()

    # Create a set of agents (exactly four)
    agent_list = [
        test_agent,
        agents.SimpleAgent(),
        # agents.RandomAgent(),
        agents.SimpleAgent(),
        
        agents.SimpleAgent()
        # agents.DockerAgent("pommerman/simple-agent", port=12345),
    ]

    print(agent_list[0])
    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFACompetition-v0', agent_list)

    tf_agent = test_agent.initialize(env)
    tf_agent.restore_model('./saved_models')

    observations = []
    inputs = []

    # Run the episodes just like OpenAI Gym
    for i_episode in range(1):
        state = env.reset()
        done = False
        while not done:
            env.render()
            actions = env.act(state)

            ppo_state = envWrapper(env, state, 0)

            actions[0] = tf_agent.act(ppo_state)
            state, reward, done, info = env.step(actions)
            # TODO Change indices of arrays to select player info.
            # observations.append({
            #     'state': env.get_json_info(), 'reward': reward, 'done': done, 'actions': actions})
            

        print('Episode {} finished'.format(i_episode))

        # save_game(i_episode, observations, info, agent_list)

    env.close()


if __name__ == '__main__':
    main()
