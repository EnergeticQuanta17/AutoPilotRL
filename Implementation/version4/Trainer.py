import os
import datetime
import gym
import json
from stable_baselines3 import PPO

def check_json_exists(filename):
    try:
        open(filename, 'r')
    except:
        main = {
            "env" : "RoadRunner-v0",
            "algo" : "PPO",
            "policy" : "MlpPolicy",
            "counter" : 0,
        }
        with open(filename, "w") as f:
            json.dump(main, f)

check_json_exists("previous_request.json")

class RLAgentTrainer:
    def __init__(self, env_name, algo):
        with open("previous_request.json", "r") as f:
            self.data = json.loads(f.read())

        self.env_name = env_name
        self.algorithm = algo
        self.counter = self.data["counter"] + 1

        self.env = gym.make(self.env_name)
        self.env.reset()

        self.data["counter"] += 1
        with open("previous_request.json", "w") as f:
            json.dump(self.data, f)
        
    def make_directories(self, models_dir, logdir):
        if(not os.path.exists(models_dir)):
            os.makedirs(models_dir)

        if(not os.path.exists(logdir)):
            os.makedirs(logdir)

    def learn(self, timestep, iterations, hyps):
        model_dir = f"model/{self.env_name}/{self.algorithm}/{self.counter}"
        logdir = f"logs/{self.env_name}/{self.algorithm}"
        self.make_directories(model_dir, logdir)

        model = PPO(**hyps)

        dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        for i in range(1, iterations+1):
            model.learn(total_timesteps=timestep, reset_num_timesteps=False, tb_log_name=str(self.counter))
            model.save(model_dir+f"/{dt}_{timestep*i}")



if(__name__=="__main__"):
    check_json_exists("previous_request.json")
