import optuna
from Trainer import *
from PPO_HypConfig import *
from Loader import *

environment_name = "CartPole-v1"
algorithm = "PPO"

timesteps = 100
iterations = 2

execution_number = 100
no_of_episodes = 2

number_of_trials = 2

def objective(trial):
    m = RLAgentTrainer(environment_name, algorithm)
    m.learn(timesteps, iterations, HypRequestHandler().optuna_next_sample(trial, m.env, "logs"))

    with open("previous_request.json", "r") as f:
        data = json.loads(f.read())

    m_load  = RLAgentLoader(environment_name, algorithm)
    return m_load.load(data['counter'], no_of_episodes)["total_return"]

if(__name__=="__main__"):
    study = optuna.create_study()
    study.optimize(objective, n_trials=number_of_trials)