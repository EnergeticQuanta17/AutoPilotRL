import plotly.offline as pyo

import logging
import sys
import json

import optuna
from sklearn import model_selection

from scipy.stats import randint, uniform

import gym

from Trainer import *
from PPO_HypConfig import *
from Loader import *
from optuna_tuner import *

all_optimizers  =[
    'optuna',
    'sklearn'
]

domain = {
    "env_name" : [env_name.id for env_name in gym.envs.registry.all()],
    "algo" : ["A2C", "DDPG", "DQN", "PPO", "SAC", "TD3"],
    "optimizer": ["optuna"],
    "sampler": ["TPESampler", "RandomSampler", "GridSampler", "PartialFixedSampler", "QMCSampler", "CmaEsSampler", "MOTPESampler", "BruteForceSampler", "NSGAIISampler"],
    "pruner": ["HyperbandPruner", "MedianPruner", "NoPruner", "PatientPruner", "PercentilePruner", "SuccessiveHalvingPruner", "ThresholdPruner"],
}

try:
    open('previous_request.json', 'r')
except:
    main = {
        "env_name" : "CartPole-v1",
        "algo" : "PPO",
        "optimizer": "optuna",
        "timesteps": 1000,
        "iterations": 10,
        "n_trials": 2,
        "counter" : 0,
        "sampler": "TPESampler",
        "pruner": "MedianPruner"
    }
    with open("previous_request.json", "w") as f:
        json.dump(main, f)

class HyperPilotRL:
    def __init__(self, env_name, algo, optimizer, timesteps, iterations, n_trials, counter, sampler=None, pruner=None, study_name="", study_directory=""):
        self.env_name = env_name
        self.algorithm = algo
        if(sampler is None):
            sampler = "TPESampler"
        if(pruner is None):
            pruner = "MedianPruner"
        self.directory = f"{optimizer}/{counter}/{env_name}/{algo}/{sampler}/{pruner}"
        self.optimizer = optimizer
        self.timestep = timesteps
        self.iterations = iterations
        self.n_trials = n_trials
        self.execution_number = counter
        self.sampler = sampler
        self.pruner = pruner

        if(self.optimizer == "optuna"):
            self.o = OptunaTuner(env_name, algo, self.directory, optimizer, timesteps, iterations, n_trials, counter, sampler, pruner)
            # o.call_optuna()
            # # o.study_summaries()
            # # o.visul()
        elif(self.optimizer == "default"):
            pass
        else:
            raise NameError("Enter optimizer is not supported.\nAvailable optimizers are", all_optimizers)

        ## optimizer = ['optuna']

    def learning_curve():
        pass

    def cutoff_learn_according_to_max_time_allocation():
        pass

    def hyp_search(self):   # for all implemented optimizers go through this to do "hyperparameter search part"
        if(self.optimizer == "optuna"):
            self.o.call_optuna()
    
    def summary(self):
        if(self.optimizer == "optuna"):
            self.o.study_summaries()
    
    def visualization(self):
        if(self.optimizer == "optuna"):
            self.o.visul()

    def get_study_data(self):
        if(self.optimizer == "optuna"):
            return self.o.return_study()

    def check_validity(self):
        if(self.optimizer == "optuna"):
            self.o.return_study()

if(__name__=="__main__"):
    with open("study_data.json", "w") as f:
        f.write("")

    #Keeping everything constant except samplers and pruners
    with open("previous_request.json", "r") as f:
        data = json.loads(f.read())
        comp_issues = []
        
        for i, j in data.items():
            if(i in ['sampler', 'pruner', 'counter', 'study_directory']):
                continue
            if(i!="env_name" and i in domain):
                print(f"{i}\n    {domain[i]}")
            while(True):
                inp = input(f"Current value of {i} is {j}. Change/Pass? ")
                if(inp==""):
                    break

                if(i in domain):
                    if(inp!="" and inp in domain[i]):
                        data[i] = inp
                        break
                else:
                    data[i] = int(inp)
                    break

        data["counter"] += 1

        for i in domain["sampler"]:
            for j in domain['pruner']:
                data['sampler'] = i
                data['pruner'] = j
                data["counter"] += 1
                data["study_name"] = f"study-{data['counter']}"
                data["study_directory"] = f'{data["optimizer"]}/{data["counter"]}/{data["env_name"]}/{data["algo"]}/{data["sampler"]}/{data["pruner"]}'
                
                # print(data)

                try:
                    print(data)
                    h = HyperPilotRL(
                        **data
                    )
                    h.hyp_search()
                    with open("study_data.json", "a") as f:
                        json.dump(data, f)
                    print("\n\n\n\n\n")
                    del h
                except Exception as e:
                    data_with_issue = data.copy()
                    data_with_issue['Exception'] = str(e)
                    comp_issues.append(data_with_issue)
                    # print(data)
                    # print("This configuration has compatibility issues!\n")
        with open("previous_request.json", "w") as f:
            json.dump(data, f)
        
        with open("comp_issues.json", "w") as f:
            json.dump(comp_issues, f)
                
        # h = HyperPilotRL(
        #     **data
        # )
        # h.hyp_search()
        # h.summary()
        # h.visualization()
        # h.get_study_data()
        