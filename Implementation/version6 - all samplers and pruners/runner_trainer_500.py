import plotly.offline as pyo

import logging
import sys
import json
import time

import optuna
from sklearn import model_selection

from scipy.stats import randint, uniform

import gym

from Trainer import *
from PPO_HypConfig import *
from Loader import *

all_optimizers  =[
    'optuna',
    'sklearn'
]

domain = {
    "env_name" : [env_name.id for env_name in gym.envs.registry.all()],
    "algo" : ["A2C", "DDPG", "DQN", "PPO", "SAC", "TD3"],
    "optimizer": ["optuna"],
    "sampler": ["BruteForceSampler", "CmaEsSampler", "GridSampler", "PartialFixedSampler", "QMCSampler", "RandomSampler", "MOTPESampler", "TPESampler", "NSGAIISampler"],
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


class OptunaTuner:
    def __init__(self, env_name, algo, directory, optimizer, ts, iterations, n_trials, counter, sampler, pruner):
        self.env_name = env_name
        self.algorithm = algo
        self.directory = directory
        self.optimizer = optimizer
        self.timestep = ts
        self.iterations = iterations
        self.n_trials = n_trials
        self.exe_number = counter

        self.sampler = sampler
        self.pruner = pruner

    def create_study_db(self):
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        study_name = f"study-{self.exe_number}"  # Unique identifier of the study.
        
        directory = '/content/drive/MyDrive/Colab_Notebooks/optuna'

        if not os.path.exists(directory):
            os.makedirs(directory)

        storage_dir = f'sqlite:///{directory}/{study_name}.db'
        return study_name, storage_dir
    
    def select_pruner(self):
        if(self.pruner is None):
            return None
        elif(self.pruner == "HyperbandPruner"):
            return optuna.pruners.HyperbandPruner(
                    min_resource=1, max_resource=10, reduction_factor=3
                )
        elif(self.pruner == "MedianPruner"):
            return optuna.pruners.MedianPruner(
                    n_startup_trials=5, n_warmup_steps=30, interval_steps=10
                )
        elif(self.pruner == "NoPruner"):
            return optuna.pruners.NopPruner()
        elif(self.pruner == "PatientPruner"):
            return optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(), patience=1)
        elif(self.pruner == "PercentilePruner"):
            return optuna.pruners.PercentilePruner(
                    25.0, n_startup_trials=5, n_warmup_steps=30, interval_steps=10
                )
        elif(self.pruner == "SuccessiveHalvingPruner"):
            return optuna.pruners.SuccessiveHalvingPruner()
        elif(self.pruner == "ThresholdPruner"):
            return optuna.pruners.ThresholdPruner(upper=1.0, lower=0.0)

    def select_sampler(self):
        if(self.sampler is None):
            return None
        if(self.sampler == "BruteForceSampler"):
            return optuna.samplers.BruteForceSampler()
        elif(self.sampler == "CmaEsSampler"):
            return optuna.samplers.CmaEsSampler()
        elif(self.sampler == "GridSampler"):
            return optuna.samplers.GridSampler()
        elif(self.sampler == "PartialFixedSampler"):
            return optuna.samplers.PartialFixedSampler()
        elif(self.sampler == "QMCSampler"):
            return optuna.samplers.QMCSampler()
        elif(self.sampler == "RandomSampler"):
            return optuna.samplers.RandomSampler()
        elif(self.sampler == "MOTPESampler"):
            return optuna.samplers.MOTPESampler()
        elif(self.sampler == "TPESampler"):
            return optuna.samplers.TPESampler()
        elif(self.sampler == "NSGAIISampler"):
            return optuna.samplers.NSGAIISampler()
        
    
    def study_loader(self):
        study_name = f"study-{self.exe_number}"  # Unique identifier of the study.
        storage_dir = f"logs/{self.directory}"
        storage_name = f"sqlite:///{storage_dir}/{study_name}.db"

        study = optuna.load_study(
            study_name=study_name,
            storage=storage_name
        )

        trials_df = study.trials_dataframe()
        print(trials_df.head())

        return study
    
    def study_summaries(self):
        study_name = f"study-{self.exe_number}"  # Unique identifier of the study.
        storage_dir = f"logs/{self.directory}"
        storage_name = f"sqlite:///{storage_dir}/{study_name}.db"

        summary = optuna.get_all_study_summaries(
            storage_name, include_best_trial=True
        )
        print(summary)
        for i,j in enumerate(summary):
            print(f"-----------STUDY-{i}-----------")
            for attribute in [attr for attr in dir(j) if not attr.startswith('__')]:
                if(attribute=="system_attrs" or attribute=="user_attrs"):
                    continue
                print(attribute, ':', getattr(j, attribute))
            print()
        pass
    
    def return_study(self, bool_store_study=True, bool_select_pruner=True, bool_select_sampler=True):
        study_name, storage_name, selected_pruner, selected_sampler = [None] * 4
        if(bool_store_study):
            study_name, storage_name = self.create_study_db()

        if(bool_select_pruner):
            selected_pruner = self.select_pruner()
        
        if(bool_select_sampler):
            selected_sampler = self.select_sampler()
        
        lie = storage_name is not None

        study = optuna.create_study(
                storage=storage_name,
                sampler=selected_sampler,
                pruner=selected_pruner,
                study_name=study_name,
                direction='maximize',
                load_if_exists=lie,
                directions=None
            )
        return study

    def call_optuna(self, store_study=True):
        study = self.return_study()
        study.optimize(self.objective, n_trials=self.n_trials)
        
    def objective(self, trial):
        m = MegaTrainer(self.env_name, self.algorithm)
        counter = m.learn(
            self.timestep,
            self.iterations,
            MegaHandler().request_next_HypConfig(trial, m.env, f"logs/{self.directory}"),
            self.directory,
            trial=trial
        )

        m_load  = MegaLoader(self.env_name, self.algorithm, self.directory)
        return m_load.load(counter, 5)["total_return"]

    def visul(self):
        study = self.study_loader()
        if(not "_optimization_history.py"):
            pyo.plot(optuna.visualization.plot_optimization_history(study))
        
        if(not "_timeline.py"):
            pyo.plot(optuna.visualization.plot_timeline(study))
        
        if(not "_slice.py"):
            pyo.plot(optuna.visualization.plot_slice(study))
        
        if(not "_pareto_front.py"): ## this is used for multi-objective study
            pyo.plot(optuna.visualization.plot_pareto_front(study))
        
        if("_param_importances.py"):
            pyo.plot(optuna.visualization.plot_param_importances(study))
        
        if(not "_parallel_coordinate.py"):
            pyo.plot(optuna.visualization.plot_parallel_coordinate(study))
            
        if(not "_intermediate_values.py"):
            pyo.plot(optuna.visualization.plot_intermediate_values(study))
        
        if(not "_edf.py"):
            pyo.plot(optuna.visualization.plot_edf(study))
            
        if(not "_contour.py"):
            pyo.plot(optuna.visualization.plot_contour(study))

class HyperPilotRL:
    def __init__(self, env_name, algo, optimizer, timesteps, iterations, n_trials, counter, sampler=None, pruner=None):
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

if(__name__=="__main__"):
#     data = {}
#     with open("previous_request.json", "r") as f:
#         data = json.loads(f.read())
        
#         for i, j in data.items():
#             if(i!="env_name" and i in domain):
#                 print(f"{i}\n    {domain[i]}")
#             while(True):
#                 inp = input(f"Current value of {i} is {j}. Change/Pass? ")
#                 if(inp==""):
#                     break

#                 if(i in domain):
#                     if(inp!="" and inp in domain[i]):
#                         data[i] = inp
#                         break
#                 else:
#                     data[i] = int(inp)
#                     break
    
#         data["counter"] += 1

#         print(data)

#         with open("previous_request.json", "w") as f:
#             json.dump(data, f)

    h = HyperPilotRL(
        env_name="CartPole-v1",
        algo="PPO",
        optimizer="optuna",
        timesteps=1000,
        iterations=500,
        n_trials=100,
        counter=100100100100,
        sampler="RandomSampler",
        pruner="HyperbandPruner"
    )

    start = time.time()
#     h = HyperPilotRL(
#         **data
#     )
    h.hyp_search()

    print("--------------------------------")
    print(time.time() - start)
    print("--------------------------------")

#     h.summary()
#     h.visualization()
#     h.get_study_data()
    
