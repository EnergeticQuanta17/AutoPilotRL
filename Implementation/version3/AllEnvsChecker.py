import pandas as pd
import time
import json
import traceback

import gym
from RLAgentBuilder import *



all_environments_latest_version = [env_name.id for env_name in gym.envs.registry.all()]

available_algorithms = ["A2C",
                        "DDPG",
                        "DQN",
                        "PPO",
                        "SAC",
                        "TD3",]
all_environments_latest_version = all_environments_latest_version[::-1]

# print(len(all_environments_latest_version))

df = pd.DataFrame(columns=["env_name", "algo", "error_name"])

start = time.time()

print("    Env_number, Time taken to check")
for index, var in enumerate(all_environments_latest_version):
	print('\t', index, '\t',time.time()-start)
	start = time.time()
	i = var
	for j in available_algorithms:
		try:
			m = RLAgent(False)
			m.third_init()
			m.second_init(i, j, "MlpPolicy")
			m.learn_and_save(1, 1)
		except Exception as e:
			error_msg = traceback.format_exc().strip().splitlines()[-1]
			df.loc[len(df)] = [i, j, error_msg]

with open("envs_with_errors.json", "w") as f:
	json.dump(df.to_json(), f)

with open("envs_with_errors.json", "r") as f:
	my_loaded_list = json.load(f)
	loaded_df = pd.read_json(my_loaded_list)

print(loaded_df['error_name'])
print(loaded_df.head())
