import json
import os
from pathlib import Path
import gym
from stable_baselines3 import PPO, A2C, DQN, DDPG, SAC, TD3
from gym.wrappers import Monitor
import time
import datetime
import random

available_algorithms = ["A2C",
                        "DDPG",
                        "DQN",
                        "PPO",
                        "SAC",
                        "TD3",]
    # On- Policy Algorithms: PPO, A2C, DQN
    # Off-Policy Algorithms: SAC, TD3, DDPG

all_environments_latest_version = [env_name.id for env_name in gym.envs.registry.all()]

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

class RLAgent:
    def __init__(self, initialize=True):
        if(not initialize):
            return
        with open("previous_request.json", "r") as f:
            data = json.loads(f.read())
    
        i = input(f"Do you want to change - (current --> {data['env']}) - Environment Name: ")
        if(i=="True" or i=="Yes"):
            while(i=="" or i=="True" or i=="Yes"):
                rsample = random.sample(all_environments_latest_version, 10)
                for j,k in enumerate(rsample):
                    print(j, k)
                i = input()
                if(not(i=="" or i=="True" or i=="Yes")):
                    envi_name = rsample[int(i)]
                    break
            print(f"Environment changed to --> {envi_name}")
        elif(i!=""):
            envi_name=i 
        else:
            envi_name = None

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

        if(self.algorithm not in available_algorithms):
            raise Exception("{self.algorithm} does not exist in the list of available algorithms: {available_algorithms}")
        
        if(self.env_name not in all_environments_latest_version):
            raise Exception("{self.env_name} does not exist in the list of available environments: {all_environments_latest_version}")

        # if(self.env_name not in available_policies):
        #     raise Exception("{self.policies} does not exist in the list of available policies: {available_algorithms}")

        self.env = gym.make(self.env_name)
        # self.env = gym.make(self.env_name)
        # self.env = Monitor(self.env, './video', force=True)
        self.env.reset()
    
    def third_init(self):
        with open("previous_request.json", "r") as f:
            data = json.loads(f.read())

        
        self.env_name = data["env"]
        self.algorithm = data["algo"]
        self.policy = data["policy"]
        self.counter = data["counter"]
        data["counter"] += 1

        with open("previous_request.json", "w") as f:
            json.dump(data, f)
    
    def details(self):
        print(f"Environment: {self.env_name}")
        print(f"Algorithm: {self.algorithm}")
        print(f"Policy: {self.policy}")

    def make_directories(self, models_dir, logdir):
        if(not os.path.exists(models_dir)):
            os.makedirs(models_dir)

        if(not os.path.exists(logdir)):
            os.makedirs(logdir)

    def create_model_given_algorithm(self, algo, policy, env, v, tbl):
        # print(algo)
        if(algo == "PPO"):
            return PPO(policy, env, verbose=0, tensorboard_log=tbl)
        elif(algo == "A2C"):
            return A2C(policy, env, verbose=0, tensorboard_log=tbl)
        elif(algo == "DQN"):
            return DQN(policy, env, verbose=0, tensorboard_log=tbl)
        elif(algo == "DDPG"):
            env.action_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(1,))
            return DDPG(policy, env, verbose=0, tensorboard_log=tbl)
        elif(algo == "SAC"):
            env.action_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(1,))
            return SAC(policy, env, verbose=0, tensorboard_log=tbl)
        elif(algo == "TD3"):
            env.action_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(1,))
            return TD3(policy, env, verbose=0, tensorboard_log=tbl)
    
    def create_model_from_PPO_hyperparameters(self, hyperparameters):
        return PPO(**hyperparameters)
    
    def learn_and_save(self, timestep, iterations):
        self.counter += 1
        model_dir = f"model/{self.env_name}/{self.algorithm}/{self.counter}"
        logdir = f"logs/{self.env_name}/{self.algorithm}"

        self.make_directories(model_dir, logdir)

        # print(self.algorithm)
        model = self.create_model_given_algorithm(algo=self.algorithm, policy=self.policy, env=self.env, v=1, tbl=logdir+"//")
        
        # hyps = PPO_HypConfig.request_next_HypConfig()
        # model = self.create_model_from_PPO_hyperparameters(hyps)
        
        TIMESTEPS = timestep

        dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # print(TIMESTEPS, iterations)
        for i in range(1, iterations+1):
            model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=str(self.counter))
            model.save(model_dir+f"/{TIMESTEPS*i}_{dt}")
        

    def load(self, no_of_episodes=1, coming_from_pyqt=False, all_files="", training_till="", idk="", env=""):
        if(not coming_from_pyqt):
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

        
        print(all_files)
        # while(not Path(model_dir)):
        #     continue
        if(coming_from_pyqt):
            model_dir = f"{idk}/{all_files[index]}"
            model = PPO.load(model_dir, env=env)
        else:
            print(all_files[index][:-3])
            model_dir = f"{model_dir}/{all_files[index]}"
            model = PPO.load(model_dir[:-4], env=self.env)

        if(coming_from_pyqt):
            env = gym.make(self.env_name)
            for ep in range(no_of_episodes):
                obs = env.reset()
                done = False
                while not done:
                    action, _states = model.predict(obs)
                    obs, rewards, done, info = env.step(action)
                    env.render()
            time.sleep(2)
            env.close()
        else:
            self.env = gym.make(self.env_name)
            for ep in range(no_of_episodes):
                obs = self.env.reset()
                done = False
                while not done:
                    action, _states = model.predict(obs)
                    obs, rewards, done, info = self.env.step(action)
                    self.env.render()
		
        
