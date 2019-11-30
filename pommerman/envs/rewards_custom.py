# TODO-USC Shape rewards here
class Rewards:
    def __init__(self):
        self.obs = None
        self.agent_list = None
        self.died_agents = 0 #num of agents dead this timestep
        self.safety_reward=None #list index is agent id:  ratio is (dist to bomb)/bomb strength, else +1 if not in danger
        self.agent_ids=None 
        self.layed_bombs=None #boolean list for if any agent layed a bomb in this timestep
        self.kick_powerup=None #boolean list if agents pickup this powerup
        self.extrabomb_powerup=None #boolean list if agents pickup this powerup
        self.bombstrength_powerup=None #boolean list if agents pickup this powerup
    def setAgentList(self, agents):
        self.agent_list = agents
        self.agent_ids=list(range(len(agents)))
    def update_state(self, obs):
        self.obs = obs
        
    def getRewards(self):
        lis = [0,0,0,0]

        for i in range(4):
            agent = self.agent_list[i]
            if(agent.is_alive): 
                lis[i] = agent.getPositionReward(agent.position)
                lis[i]+=(int(self.layed_bombs[i])*0.3)

            

        # for i in range(0,4):
        #     if self.agent_list[i].is_alive:
        #         lis[i] = self.died_agents

        self.died_agents = 0
        return lis

