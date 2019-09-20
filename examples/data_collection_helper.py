import os
import json

DATA_DIR = 'train_data\\'

def count_dir(save):
    folders = 0

    for _, dirnames, _ in os.walk(DATA_DIR):
        folders += len(dirnames)

    if save:
        return "{0:0=5d}".format(folders)
    else:
        return "{0:0=5d}".format(folders - 1)

def save_game(ep_count, observations, info, agent_list):
    dir_name = count_dir(True)
    os.mkdir(DATA_DIR + dir_name)
    with open('train_data\\' + dir_name + '\\' + 'ep_' + "{0:0=6d}".format(ep_count) + '.txt', 'w') as filehandle:
        # filehandle.write(str(observations))
        filehandle.write(json.dumps({
            'agents': [str(agent) for agent in agent_list],
            "result": {
                "name": info['result'].name,
                "id": info['result'].value
            },
            'observations': observations
        }, sort_keys=True, indent=4))

def load_game():
    dir_name = count_dir(False)
    with open('train_data\\' + dir_name + '\\' + 'ep_' + "{0:0=6d}".format(0) + '.txt', 'r') as filehandle:
        return json.loads(filehandle.read())