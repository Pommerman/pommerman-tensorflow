# TODO-USC Shape rewards here
class Rewards:
    def __init__(self):
        self.obs = None
        self.agent_list = None
        self.died_agents = 0
        self.safety_reward=None
        self.agent_ids=None
    def setAgentList(self, agents):
        self.agent_list = agents
        self.agent_ids=list(range(len(agents)))
    def update_state(self, obs):
        self.obs = obs

    def agent_died(self, agent):
        self.died_agents+=1
        
        
    #safety reward of each agent:
    #ratio is (dist to bomb)/bomb strength, else +1 if not in danger
    def get_safety_reward(self,safety_reward: list):
        self.safety_reward=safety_reward
    
    def getRewards(self):
        lis = [0,0,0,0]
        #add reward for when opponents die
        for i in range(0,4):
            if self.agent_list[i].is_alive:
                lis[i] = self.died_agents
                lis[i]+=self.safety_reward[i] #add reward for safety of current location for this timestep
        

        self.died_agents = 0
        return lis 

