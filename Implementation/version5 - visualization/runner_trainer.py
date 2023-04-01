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

all_optimizers  =[
    'optuna',
    'sklearn'
]

class Dummy:
    def __init__(self):
        pass

    def fit(self):
        pass

class OptunaTuner:
    def __init__(self, env_name, algo, directory, optimizer, ts, iterations, n_trials, counter):
        self.env_name = env_name
        self.algorithm = algo
        self.directory = directory
        self.optimizer = optimizer
        self.timestep = ts
        self.iterations = iterations
        self.n_trials = n_trials
        self.exe_number = counter

    def create_study_db(self):
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        study_name = f"study-{self.exe_number}"  # Unique identifier of the study.
        
        storage_dir = f"logs/{self.directory}"
        if(not os.path.exists(storage_dir)):
            os.makedirs(storage_dir)
        storage_name = f"sqlite:///{storage_dir}/{study_name}.db"

        return study_name, storage_name
    
    def select_pruner(self):
        return None

    def select_sampler(self):
        return None
    
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
        return m_load.load(counter, self.n_trials)["total_return"]

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
        
        if(not "_param_importances.py"):
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
            # o.call_optuna()
            # o.study_summaries()
            o.visul()
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
        iterations=1,
        n_trials=10,
        counter=261030
    )

    