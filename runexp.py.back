import os
import copy
from config import DIC_AGENTS, DIC_ENVS
import time
import numpy as np
import sys
from multiprocessing import Process, Pool
import tensorflow as tf
from tensorflow.compat.v1.keras import backend as K

class Generator:
    def __init__(self, cnt_round, cnt_gen, dic_path, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, best_round=None):

        self.cnt_round = cnt_round
        self.cnt_gen = cnt_gen
        self.dic_exp_conf = dic_exp_conf
        self.dic_path = dic_path
        self.dic_agent_conf = copy.deepcopy(dic_agent_conf)
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.agents = [None]*dic_traffic_env_conf['NUM_AGENTS']

        if self.dic_exp_conf["PRETRAIN"]:
            self.path_to_log = os.path.join(self.dic_path["PATH_TO_PRETRAIN_WORK_DIRECTORY"], "train_round",
                                            "round_" + str(self.cnt_round), "generator_" + str(self.cnt_gen))
        else:
            self.path_to_log = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round", "round_"+str(self.cnt_round), "generator_"+str(self.cnt_gen))
        if not os.path.exists(self.path_to_log):
            os.makedirs(self.path_to_log)  

        self.env = DIC_ENVS[dic_traffic_env_conf["SIMULATOR_TYPE"]](
                              path_to_log = self.path_to_log,
                              path_to_work_directory = self.dic_path["PATH_TO_WORK_DIRECTORY"],
                              dic_traffic_env_conf = self.dic_traffic_env_conf)        
        #self.num_actions = len(self.env.DIC_PHASE_MAP) - 1
        #self.num_actions = len(self.env.list_intersection[0].DIC_PHASE_MAP) - 1
        self.env.reset()
        self.num_actions = len(self.env.list_intersection[0].DIC_PHASE_MAP) - 1
        # every generator's output
        # generator for pretraining
        # Todo pretrain with intersection_id
        if self.dic_exp_conf["PRETRAIN"]:
            self.agent_name = self.dic_exp_conf["PRETRAIN_MODEL_NAME"]
            self.agent = DIC_AGENTS[agent_name](
                dic_agent_conf=self.dic_agent_conf,
                dic_traffic_env_conf=self.dic_traffic_env_conf,
                dic_path=self.dic_path,
                cnt_round=self.cnt_round,
                best_round=best_round,
                intersection_id=str(i),
                num_actions=self.num_actions
            )

        else:

            start_time = time.time()

            for i in range(dic_traffic_env_conf['NUM_AGENTS']):
                agent_name = self.dic_exp_conf["MODEL_NAME"]
                #the CoLight_Signal needs to know the lane adj in advance, from environment's intersection list
                if agent_name=='CoLight_Signal':
                    agent = DIC_AGENTS[agent_name](
                        dic_agent_conf=self.dic_agent_conf,
                        dic_traffic_env_conf=self.dic_traffic_env_conf,
                        dic_path=self.dic_path,
                        cnt_round=self.cnt_round, 
                        best_round=best_round,
                        inter_info=self.env.list_intersection,
                        intersection_id=str(i)
                    )      
                else:              
                    agent = DIC_AGENTS[agent_name](
                        dic_agent_conf=self.dic_agent_conf,
                        dic_traffic_env_conf=self.dic_traffic_env_conf,
                        dic_path=self.dic_path,
                        cnt_round=self.cnt_round, 
                        best_round=best_round,
                        intersection_id=str(i)
                    )
                self.agents[i] = agent
        sess = K.get_session()
        sess.run(tf.compat.v1.global_variables_initializer())        
        print("Create intersection agent time: ", time.time()-start_time)

    def wrap_state(state_raw):
        wrapped = []
        for s in state_raw:
            raw_parts = []
            for key in list_state_features:   # same keys used in reset
                arr = np.array(s[key]).reshape(-1)
                raw_parts.append(arr)

            feature = np.concatenate(raw_parts, axis=0)

            # pad/truncate to 32
            if feature.size < 32:
                feature = np.pad(feature, (0, 32 - feature.size))
            elif feature.size > 32:
                feature = feature[:32]

            wrapped.append({
                "feature": feature.astype(np.float32),
                "adjacency_matrix": np.array(s["adjacency_matrix"])
            })
        return wrapped

    def generate(self):
        print("ENV CLASS:", type(self.env))
        print("ENV MODULE:", self.env.__class__.__module__)
        reset_env_start_time = time.time()
        done = False
        state = self.env.reset()
        print("RESET FEATURE SHAPES:",[len(s["feature"]) for s in state])
        step_num = 0
        reset_env_time = time.time() - reset_env_start_time

        running_start_time = time.time()

        max_steps = int(self.dic_exp_conf["RUN_COUNTS"] /
                        self.dic_traffic_env_conf["MIN_ACTION_TIME"])
        num_intersections = self.dic_traffic_env_conf["NUM_INTERSECTIONS"]
        print("num_actions (agent)::", self.agents[0].num_actions)
        while not done and step_num < max_steps:

            step_start_time = time.time()

            # ---------------------------------------------------------
            # 1. Convert anon_env state → CoLight state (ONE TIME)
            # ---------------------------------------------------------
            if isinstance(state, list):   # anon_env format
                features = []
                adjs = []
                num_agents = self.dic_traffic_env_conf["NUM_INTERSECTIONS"]
                #num_neighbors = len(state[0]["adjacency_matrix"])
                num_neighbors = self.dic_traffic_env_conf["NUM_NEIGHBORS"]

                for agent_id in range(num_agents):
                    s = state[agent_id]

                    # s["feature"] already exists and is length 32
                    features.append(np.array(s["feature"], dtype=np.float32))
                    neighbor_ids = s["adjacency_matrix"]

                    adj_agent = np.zeros((num_neighbors, num_agents), dtype=np.float32)

                    for k in range(num_neighbors):
                        if k >= len(neighbor_ids):
                            continue

                        nid = neighbor_ids[k]

                        if nid is None:
                            continue

                        try:
                            nid = int(nid)
                        except:
                            continue

                        if nid < 0 or nid >= num_agents:
                            continue

                        adj_agent[k, nid] = 1.0
                    '''neighbor_ids = s["adjacency_matrix"]
                    adj_agent = np.zeros((num_neighbors, num_agents), dtype=np.float32)

                    for k, nid in enumerate(neighbor_ids):
                        if nid is None:
                            continue
                        try:
                            nid = int(nid)
                        except:
                            continue
                        if nid < 0 or nid >= num_agents:
                            continue
                        adj_agent[k, nid] = 1.0
                    '''   
                    adjs.append(adj_agent)

                one_state = {
                    "feature": np.stack(features, axis=0),          # (N, 32)
                    "adjacency_matrix": np.array(adjs, dtype=np.float32)  # (N, K, N)
                }
            else:
                one_state = state

            # ---------------------------------------------------------
            # 2. Choose actions for ALL intersections
            # ---------------------------------------------------------
            #print("num_phases (env):", len(self.env.__class__.DIC_PHASE_MAP))
            print("one_state feature shape:", one_state["feature"].shape)
            print("one_state adj shape:", one_state["adjacency_matrix"].shape)

            actions, _ = self.agents[0].choose_action(step_num, one_state)

            # actions is now a 1D array of length num_agents
            assert len(actions) == num_agents, f"Got {len(actions)} actions for {num_agents} intersections" 
            # ---------------------------------------------------------
            # 3. Environment step
            # ---------------------------------------------------------
            #next_state, reward, done, _ = self.env.step(action_list)
            next_state, reward, done, _ = self.env.step(actions)

            # build next_one_state exactly like one_state
            features_next = []
            adjs_next = []
            num_agents = self.dic_traffic_env_conf["NUM_INTERSECTIONS"]

            #num_neighbors = len(next_state[0]["adjacency_matrix"])
            num_neighbors = self.dic_traffic_env_conf["NUM_NEIGHBORS"]
            for agent_id in range(num_agents):
                s_next = next_state[agent_id]

                features_next.append(np.array(s_next["feature"], dtype=np.float32))

                neighbor_ids = s_next["adjacency_matrix"]
                adj_agent_next = np.zeros((num_neighbors, num_agents), dtype=np.float32)

                for k in range(num_neighbors):
                    if k >= len(neighbor_ids):
                        continue

                    nid = neighbor_ids[k]

                    if nid is None:
                        continue

                    try:
                        nid = int(nid)
                    except:
                        continue

                    if nid < 0 or nid >= num_agents:
                        continue

                    adj_agent_next[k, nid] = 1.0

                adjs_next.append(adj_agent_next)

            next_one_state = {
                "feature": np.stack(features_next, axis=0),
                "adjacency_matrix": np.array(adjs_next, dtype=np.float32)
            }

            self.agents[0].remember(one_state, actions, reward, next_one_state)

            # ---------------------------------------------------------
            # 4. Training (per intersection)
            # ---------------------------------------------------------
            # Store transition for the single CoLight agent

            # Train the single agent
            self.agents[0].train_network()
            # ---------------------------------------------------------
            # 5. Logging + update state
            # ---------------------------------------------------------
            print("time: {0}, running_time: {1}".format(
                self.env.get_current_time() -
                self.dic_traffic_env_conf["MIN_ACTION_TIME"],
                time.time() - step_start_time
            ))

            state = next_one_state
            step_num += 1

        # ---------------------------------------------------------
        # 6. End of episode logging
        # ---------------------------------------------------------
        running_time = time.time() - running_start_time

        log_start_time = time.time()
        print("start logging")
        self.env.bulk_log_multi_process()
        log_time = time.time() - log_start_time

        self.env.end_sumo()
        print("reset_env_time: ", reset_env_time)
        print("running_time: ", running_time)
        print("log_time: ", log_time)
       
