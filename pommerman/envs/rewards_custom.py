# TODO-USC Shape rewards here
class Rewards:
    def __init__(self):
        self.obs = None
        self.agent_list = None
        self.died_agents = 0

    def setAgentList(self, agents):
        self.agent_list = agents

    def update_state(self, obs):
        self.obs = obs

    def agent_died(self, agent):
        self.died_agents+=1

    def getRewards(self):
        lis = [0,0,0,0]

        for i in range(0,4):
            if self.agent_list[i].is_alive:
                lis[i] = self.died_agents

        self.died_agents = 0
        return lis 

