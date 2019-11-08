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
        for i in range(0,4):
            if self.agent_list[i].is_alive:
                lis[i] = self.died_agents  #add reward for when opponents die
                lis[i]+=(self.safety_reward[i]*0.001) #add reward for safety of current location for this timestep, scaled to .01 so its less reward than laying a bomb
                lis[i]+=(int(self.layed_bombs[i])*0.5) #add reward if agent layed a bomb this timestep, but it should be less than if you kill an opponent
                lis[i]+=(int(self.kick_powerup[i])*.1)
                lis[i]+=(int(self.extrabomb_powerup[i])*.1)
                lis[i]+=(int(self.bombstrength_powerup[i])*.1)
        return lis 

