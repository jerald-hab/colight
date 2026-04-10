print(">>> LOADING NEW GENERATOR FILE:", __file__)
import os
import copy 
import time
import numpy as np
import sys
from multiprocessing import Process, Pool
from colight_pkg.config import DIC_AGENTS, DIC_ENVS
# from colight_pkg.anon_env import AnonEnv
# from colight_pkg.agent import Agent   # if used
# from colight_pkg.model_pool import MODEL_POOL  # if used

class Generator:
    def __init__(self, cnt_round, cnt_gen, dic_path, dic_exp_conf,
             dic_agent_conf, dic_traffic_env_conf, best_round=None):

        print(">>> ENTER Generator.__init__", flush=True)
        print(">>> dic_traffic_env_conf:", dic_traffic_env_conf, flush=True)
        print(">>> DIC_PATH:", dic_path, flush=True)
        import os, shutil, copy

        # FIRST: assign attributes
        self.cnt_round = cnt_round
        self.cnt_gen = cnt_gen
        self.dic_path = dic_path
        self.dic_exp_conf = dic_exp_conf
        self.dic_agent_conf = copy.deepcopy(dic_agent_conf)
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.best_round = best_round

        print(">>> RAW ROADNET_FILE:", self.dic_traffic_env_conf["ROADNET_FILE"], flush=True)
        print(">>> RAW TRAFFIC_FILE:", self.dic_traffic_env_conf["TRAFFIC_FILE"], flush=True) 
        dataset_dir = os.path.abspath(self.dic_path["PATH_TO_DATA"])

        roadnet_path = os.path.abspath(
            os.path.join(dataset_dir, self.dic_traffic_env_conf["ROADNET_FILE"])
        )
        traffic_path = os.path.abspath(
            os.path.join(dataset_dir, self.dic_traffic_env_conf["TRAFFIC_FILE"])
        )
        self.dic_traffic_env_conf["ROADNET_FILE"] = roadnet_path
        self.dic_traffic_env_conf["TRAFFIC_FILE"] = traffic_path
        print(">>> DATASET DIR:", dataset_dir, flush=True)
        print(">>> RESOLVED ROADNET_FILE:", roadnet_path, flush=True)
        print(">>> RESOLVED TRAFFIC_FILE:", traffic_path, flush=True)
        print(">>> ROADNET EXISTS?", os.path.exists(roadnet_path), flush=True)
        print(">>> TRAFFIC EXISTS?", os.path.exists(traffic_path), flush=True) 

        # Each generator gets its own isolated work directory
        self.path_to_work_directory = os.path.join(
            dic_path["PATH_TO_WORK_DIRECTORY"],
            "train_round",
            f"round_{cnt_round}",
            f"generator_{cnt_gen}"
        )
        os.makedirs(self.path_to_work_directory, exist_ok=True)

        # Copy dataset files into THIS generator's directory
        dst_roadnet = os.path.join(self.path_to_work_directory, "roadnet_6_6.json")
        dst_flow = os.path.join(self.path_to_work_directory, "anon_6_6_300_0.3_bi.json")

        if not os.path.exists(dst_roadnet):
            shutil.copy(roadnet_path, dst_roadnet)

        if not os.path.exists(dst_flow):
            shutil.copy(traffic_path, dst_flow)

        print("ROADNET:", self.dic_traffic_env_conf["ROADNET_FILE"])
        print("FLOW:", self.dic_traffic_env_conf["TRAFFIC_FILE"])
        print("WORKDIR:", self.path_to_work_directory)

        # Log directory should also be generator-specific
        self.path_to_log = self.path_to_work_directory

        # Create environment using the generator-specific directory
        self.env = DIC_ENVS[self.dic_traffic_env_conf["SIMULATOR_TYPE"]](
            path_to_log=self.path_to_log,
            path_to_work_directory=self.path_to_work_directory,
            dic_traffic_env_conf=self.dic_traffic_env_conf
        )

        self.env.reset()
        print("Lane count:", len(self.env.list_intersection[0].list_entering_lanes))
        print("=== PHASE MAPS FOR ALL INTERSECTIONS ===")
        for idx, inter in enumerate(self.env.list_intersection):
            print(f"Intersection {idx}: {inter.DIC_PHASE_MAP}")
        print("========================================")
        # 2. Extract the real phase map from the environment
        phase_map = self.env.list_intersection[0].DIC_PHASE_MAP


        self.valid_actions = sorted(list(phase_map.keys()))
        
        self.num_actions = len(self.valid_actions)
        # Number of actions = number of phases

        # Number of intersections in the grid
        self.num_inters = self.dic_traffic_env_conf["NUM_ROW"] * self.dic_traffic_env_conf["NUM_COL"]

        # Create ONE shared agent
        start_time = time.time()
        agent_name = self.dic_exp_conf["MODEL_NAME"]

        self.agent = DIC_AGENTS[agent_name](
            dic_agent_conf=self.dic_agent_conf,
            dic_traffic_env_conf=self.dic_traffic_env_conf,
            dic_path=self.dic_path,
            cnt_round=self.cnt_round,
            best_round=best_round,
            intersection_id="0",
            DIC_PHASE_MAP=phase_map
        )

        # Shared agent list: same agent for all intersections
        self.agents = [self.agent] * self.num_inters

        print("Create intersection agent time:", time.time() - start_time)

        # for i in range(1): #dic_traffic_env_conf['NUM_AGENTS'] only 1 agent even though multi agent envt
        #     agent_name = self.dic_exp_conf["MODEL_NAME"]
        #     #the CoLight_Signal needs to know the lane adj in advance, from environment's intersection list
        #     if agent_name=='CoLight_Signal':
        #         agent = DIC_AGENTS[agent_name](
        #             dic_agent_conf=self.dic_agent_conf,
        #             dic_traffic_env_conf=self.dic_traffic_env_conf,
        #             dic_path=self.dic_path,
        #             cnt_round=self.cnt_round, 
        #             best_round=best_round,
        #             inter_info=self.env.list_intersection,
        #             intersection_id=str(i)
        #         )      
        #     else:              
        #         agent = DIC_AGENTS[agent_name](
        #             dic_agent_conf=self.dic_agent_conf,
        #             dic_traffic_env_conf=self.dic_traffic_env_conf,
        #             dic_path=self.dic_path,
        #             cnt_round=self.cnt_round, 
        #             best_round=best_round,
        #             intersection_id=str(i)
        #         )
        #     self.agents[i] = agent
        # print("Create intersection agent time: ", time.time()-start_time)
 
    # def generate(self):

    #     reset_env_start_time = time.time()
    #     done = False
    #     state = self.env.reset()
    #     step_num = 0
    #     reset_env_time = time.time() - reset_env_start_time

    #     running_start_time = time.time()

    #     while not done and step_num < int(self.dic_exp_conf["RUN_COUNTS"]/self.dic_traffic_env_conf["MIN_ACTION_TIME"]):
    #         action_list = []
    #         step_start_time = time.time()

    #         for i in range(self.dic_traffic_env_conf["NUM_AGENTS"]): 
    #             if self.dic_exp_conf["MODEL_NAME"] in ["CoLight","GCN", "SimpleDQNOne"]:
    #                 one_state = state
    #                 if self.dic_exp_conf["MODEL_NAME"] == 'CoLight':
    #                     if self.cnt_round == 0:
    #                         # Warm-up: random or fixed-time action
    #                         action = np.random.randint(self.num_actions)
    #                     else:
    #                         action, _ = self.agents[i].choose_action(step_num, one_state)    
    #                     # action, _ = self.agents[i].choose_action(step_num, one_state) jj
    #                 elif self.dic_exp_conf["MODEL_NAME"] == 'GCN':
    #                     action = self.agents[i].choose_action(step_num, one_state)
    #                 else: # simpleDQNOne
    #                     if True:
    #                         action = self.agents[i].choose_action(step_num, one_state)
    #                     else:
    #                         action = self.agents[i].choose_action_separate(step_num, one_state)
    #                 action_list = action
    #             else:
    #                 one_state = state[i]
    #                 action = self.agents[i].choose_action(step_num, one_state)
    #                 action_list.append(action)

    #         next_state, reward, done, _ = self.env.step(action_list)

    #         print("time: {0}, running_time: {1}".format(self.env.get_current_time()-self.dic_traffic_env_conf["MIN_ACTION_TIME"],
    #                                                     time.time()-step_start_time))
    #         state = next_state
    #         step_num += 1
    #     running_time = time.time() - running_start_time

    #     log_start_time = time.time()
    #     print("start logging")
    #     self.env.bulk_log_multi_process()
    #     log_time = time.time() - log_start_time

    #     self.env.end_sumo()
    def generate(self):
        reset_env_start_time = time.time()
        done = False
        state = self.env.reset()
        step_num = 0
        reset_env_time = time.time() - reset_env_start_time

        running_start_time = time.time()

        # Number of intersections in the grid
        num_inters = self.num_inters ##dic_traffic_env_conf["NUM_INTERSECTIONS"]

        while not done and step_num < int(self.dic_exp_conf["RUN_COUNTS"] / self.dic_traffic_env_conf["MIN_ACTION_TIME"]):

            step_start_time = time.time()
            action_list = []

            # -------------------------------
            #   ACTION SELECTION PER INTERSECTION
            # -------------------------------
            for inter_id in range(num_inters):

                # CoLight / STGAT use ONE shared agent
                if self.dic_exp_conf["MODEL_NAME"] in ["CoLight", "STGAT", "GCN"]:

                    if self.cnt_round == 0:
                        # warm-up: random action for each intersection
                        action_list = [np.random.choice(self.valid_actions) for _ in range(num_inters)]
                    else:
                        action_list, _ = self.agents[0].choose_action(step_num, state)
                        # ensure Python ints
                        action_list = [int(a) for a in action_list]

                else:
                    # other models: one agent per intersection
                    action_list = []
                    for inter_id in range(num_inters):
                        one_state = state[inter_id]
                        action = self.agents[inter_id].choose_action(step_num, one_state)
                        action_list.append(int(action))


            # -------------------------------
            #   STEP ENVIRONMENT
            # -------------------------------
            next_state, reward, done, _ = self.env.step(action_list)

            # print("time: {0}, running_time: {1}".format(
            #     self.env.get_current_time() - self.dic_traffic_env_conf["MIN_ACTION_TIME"],
            #     time.time() - step_start_time
            # ))

            state = next_state
            step_num += 1

        running_time = time.time() - running_start_time

        # -------------------------------
        #   LOGGING
        # -------------------------------
        log_start_time = time.time()
        print("start logging")
        self.env.bulk_log_multi_process()
        log_time = time.time() - log_start_time

        self.env.end_sumo()
        print("reset_env_time: ", reset_env_time)
        print("running_time: ", running_time)
        print("log_time: ", log_time)
