import sys
sys.path.append("..")
import random
from colight_pkg.agent import Agent


class RandomAgent(Agent):

    def __init__(self, dic_agent_conf, dic_sumo_env_conf, dic_path, cnt_round, best_round=None):

        super(RandomAgent, self).__init__(dic_agent_conf, dic_sumo_env_conf, dic_path)



    def choose_action(self, count, state):
        ''' choose the best action for current state '''

        action = random.randrange(self.dic_sumo_env_conf["NUM_PHASES"])

        return action

