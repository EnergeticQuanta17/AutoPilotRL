import optuna
from Trainer import *
from PPO_HypConfig import *
from Loader import *

def objective(trial):
    m = MegaTrainer("CartPole-v1", "PPO")
    m.learn(10, 1, MegaHandler().request_next_HypConfig(trial, m.env, ""))

    m_load  = MegaLoader("CartPole-v1", "PPO")
    return m_load.load(214, 2)["total_return"]

if(__name__=="__main__"):
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)