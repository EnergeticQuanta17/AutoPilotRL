import plotly.offline as pyo

import logging
import sys

import optuna
from sklearn import model_selection

from scipy.stats import randint, uniform

import gym

from Trainer import *
from PPO_HypConfig import *
from Loader import *
from optuna_tuner import OptunaTuner

all_optimizers  =[
    'optuna',
    'sklearn'
]

class HyperPilotRL:
    def __init__(self, env_name, algo, optimizer, timesteps, iterations, n_trials, counter):
        self.env_name = env_name
        self.algorithm = algo
        self.directory = f"{optimizer}/{counter}/{env_name}/{algo}"
        self.optimizer = optimizer
        self.timestep = timesteps
        self.iterations = iterations
        self.n_trials = n_trials
        self.execution_number = counter

        if(self.optimizer == "optuna"):
            o = OptunaTuner(env_name, algo, self.directory, optimizer, timesteps, iterations, n_trials, counter)
            o.call_optuna()
            # o.study_summaries()
            # o.visul()
        elif(self.optimizer == "default"):
            pass

        else:
            raise NameError("Enter optimizer is not supported.\nAvailable optimizers are", all_optimizers)

        ## optimizer = ['optuna']

    def learning_curve():
        pass

    def cutoff_learn_according_to_max_time_allocation():
        pass

if(__name__=="__main__"):
    h = HyperPilotRL(
        "CartPole-v1",
        "PPO",
        "optuna",
        timesteps=1000,
        iterations=10,
        n_trials=2,
        counter=1026
    )