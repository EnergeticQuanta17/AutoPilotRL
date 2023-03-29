import gym
from MegaD import *

all_environments_latest_version = [env_name.id for env_name in gym.envs.registry.all()]
available_algorithms = ["A2C",
                        "DDPG",
                        "DQN",
                        "PPO",
                        "SAC",
                        "TD3",]
all_environments_latest_version = all_environments_latest_version[::-1]

print(len(all_environments_latest_version))

available_algorithms = ["A2C"]

for i in all_environments_latest_version:
	# print("!!!!!!!!!", i, "!!!!!!!!!")
	# with open("envs_with_errors.txt", "a") as f:
	# 	to_store = f"-----------------------------\n{i}\n-----------------------------"
	# 	f.write(to_store + "\n")
	for j in available_algorithms:
		try:
			m = MegaD26(False)
			m.third_init()
			m.second_init(i, j, "MlpPolicy")
			m.learn_and_save(10, 1)
		except Exception as e:
			to_store = i +" " + j +" " + "MlpPolicy" +" " + type(e).__name__
			print(to_store)
			with open("envs_with_errors.txt", "a") as f:
				f.write(to_store + "\n")
	# with open("envs_with_errors.txt", "a") as f:
	# 	f.write("\n\n")