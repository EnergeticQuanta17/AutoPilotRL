import optuna
import os
import sys
import logging
from time import sleep
import plotly.offline as pyo

from Trainer import *
from Loader import *
from PPO_HypConfig import *

class OptunaTuner:
    def __init__(self, env_name, algo, directory, optimizer, ts, iterations, n_trials, counter, req_plots):
        self.env_name = env_name
        self.algorithm = algo
        self.directory = directory
        self.optimizer = optimizer
        self.timestep = ts
        self.iterations = iterations
        self.n_trials = n_trials
        self.exe_number = counter
        self.plots = req_plots

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
            HypRequestHandler().optuna_next_sample(trial, m.env, f"logs/{self.directory}"),
            self.directory,
            trial=trial
        )

        m_load  = RLAgentLoader(self.env_name, self.algorithm, self.directory)
        return m_load.load(counter, self.n_trials)["total_return"]

    def visul(self):
        index = -1
        study = self.study_loader()

        index += 1
        if(self.plots[index][1]):
            pyo.plot(optuna.visualization.plot_optimization_history(study))
            sleep(0.5)
        
        index += 1
        if(self.plots[index][1]):
            pyo.plot(optuna.visualization.plot_timeline(study))
            sleep(0.5)
        
        index += 1
        if(self.plots[index][1]):
            pyo.plot(optuna.visualization.plot_slice(study))
            sleep(0.5)
        
        index += 1
        if(self.plots[index][1]):
            pyo.plot(optuna.visualization.plot_pareto_front(study))
            sleep(0.5)
        
        index += 1
        if(self.plots[index][1]):
            pyo.plot(optuna.visualization.plot_param_importances(study))
            sleep(0.5)
        
        index += 1
        if(self.plots[index][1]):
            pyo.plot(optuna.visualization.plot_parallel_coordinate(study))
            sleep(0.5)
            
        index += 1
        if(self.plots[index][1]):
            pyo.plot(optuna.visualization.plot_intermediate_values(study))
            sleep(0.5)
        
        index += 1
        if(self.plots[index][1]):
            pyo.plot(optuna.visualization.plot_edf(study))
            sleep(0.5)
            
        index += 1
        if(self.plots[index][1]):
            pyo.plot(optuna.visualization.plot_contour(study))
            sleep(0.5)