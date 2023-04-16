from Trainer import *
from PPO_HypConfig import *
from Loader import *
from OptunaHypTuner import OptunaTuner

all_optimizers  =[
    'optuna',
    'sklearn'
]

class HyperPilotRL:
    def __init__(self, env_name, algo, optimizer, timesteps, iterations, n_trials, counter, req_plots):
        self.env_name = env_name
        self.algorithm = algo
        self.directory = f"{optimizer}/{counter}/{env_name}/{algo}"
        self.optimizer = optimizer
        self.timestep = timesteps
        self.iterations = iterations
        self.n_trials = n_trials
        self.execution_number = counter
        self.plots = req_plots

        if(self.optimizer == "optuna"):
           self.optuna()

        elif(self.optimizer == "default"):
            pass

        else:
            raise NameError("Enter optimizer is not supported.\nAvailable optimizers are", all_optimizers)

        ## optimizer = ['optuna']

    def optuna(self.):
        o = OptunaTuner(self.env_name, self.algo, self.directory, self.optimizer, self.timesteps, self.iterations, self.n_trials, self.counter, self.req_plots)
        o.call_optuna()
        o.study_summaries()
        o.visul()

    def cutoff_learn_according_to_max_time_allocation():
        pass

if(__name__=="__main__"):
    plots = [
        ('Optimization History', True),
        ('Timeline', True),
        ('Slice', True),
        ('Pareto Font', False),
        ('Params Importances', True),
        ('Parallel Coordinates', True),
        ('Intermediate Values', True),
        ('Empirical Distribution Function', True),
        ('Contour', True),
    ]

    params = {
        'env_name': "CartPole-v1",
        'algo': "PPO",
        'optimizer': "optuna",
        'timesteps': 100,
        'iterations': 1,
        'n_trials': 2,
        'counter': 10,
        'req_plots': plots
    }

    h = HyperPilotRL(
       **params
    )