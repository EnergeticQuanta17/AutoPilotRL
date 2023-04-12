import json
import os
from pathlib import Path
import gym
from stable_baselines3 import PPO
import time
import datetime

## Use previous values unless otherwise requested to change
## make dictionary to send to PPO(...) --- to dots
## parameterize everything
## parallelize load and save
## estimate the amount of time requrired given the time_steps_done , to wait in load method
## setup initial json file
## automatically choose which zip file to open in load()
## include dt in file name
# default parameter values are not working properly -- mayve use none and assign default in init
## add checks if new value of algorithm matches available
## check if folder exists in load()

try:
    open('previous_request.json', 'r')
except:
    main = {
        "env" : "CartPole-v1",
        "algo" : "PPO",
        "policy" : "MlpPolicy",
        "counter" : 0,
    }
    with open("previous_request.json", "w") as f:
        json.dump(main, f)

class MegaD26:
    def __init__(self):
        with open("previous_request.json", "r") as f:
            data = json.loads(f.read())
    
        i = input(f"Do you want to change - (current --> {data['env']}) - Environment Name: ")
        envi_name = i if (i != "") else None
        i = input(f"Do you want to change - (current --> {data['algo']}) - Algorithm: ")
        algorithm = i if (i != "") else None
        i = input(f"Do you want to change - (current --> {data['policy']}) - Policy: ")
        poli = i if (i != "") else None
        
        self.second_init(envi_name, algorithm, poli)
        
    def second_init(self, env_name=None, algo=None, policy=None):
        with open("previous_request.json", "r") as f:
            data = json.loads(f.read())
        
        self.env_name = env_name
        self.algorithm = algo
        self.policy = policy
        if(env_name == None):
            self.env_name = data["env"]
        else:
            data["env"] = env_name
        if(algo == None):
            self.algorithm = data["algo"]
        else:
            data["algo"] = algo
        if(policy == None):
            self.policy = data["policy"]
        else:
            data["policy"] = policy
        
        self.counter = data["counter"]
        data["counter"] += 1

        with open("previous_request.json", "w") as f:
            json.dump(data, f)


        self.env = gym.make(self.env_name)
        self.env.reset()

    def make_directories(self, models_dir, logdir):
        if(not os.path.exists(models_dir)):
            os.makedirs(models_dir)

        if(not os.path.exists(logdir)):
            os.makedirs(logdir)

    def learn_and_save(self, timestep, iterations):
        self.counter += 1
        model_dir = f"model/{self.env_name}/{self.algorithm}/{self.counter}"
        logdir = f"logs/{self.env_name}/{self.algorithm}"

        self.make_directories(model_dir, logdir)

        model = PPO(self.policy, self.env, verbose=1, tensorboard_log=logdir+"//")

        TIMESTEPS = timestep

        dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        print(TIMESTEPS, iterations)
        for i in range(1, iterations+1):
            print("------------------------------------------------------------")
            model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=str(self.counter))
            model.save(model_dir+f"/{TIMESTEPS*i}_{dt}")
        

    def load(self, no_of_episodes=5):
        print("Choose the model to show output on, among the following: ")
        model_dir = f"model/{self.env_name}/{self.algorithm}"
        print(os.listdir(model_dir))
        print(model_dir)
        model_no = input("Enter execution number: ")
        all_files = os.listdir(f"{model_dir}/{model_no}")
        print(all_files)
        training_till = input("Select the model: ")+"_"
        
        try:
            index = all_files.index(next(s for s in all_files if training_till in s))
            print(index)
        except StopIteration:
            print(f"{training_till} is not a substring of any string in the list")
            exit()

        model_dir = f"{model_dir}/{model_no}/{all_files[index]}"
        # while(not Path(model_dir)):
        #     continue

        model = PPO.load(model_dir, env=self.env)

        for ep in range(no_of_episodes):
            obs = self.env.reset()
            done = False
            while not done:
                action, _states = model.predict(obs)
                obs, rewards, done, info = self.env.step(action)
                self.env.render()