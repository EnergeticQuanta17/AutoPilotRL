from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
print("done")
tune.run(PPOTrainer, 
    config={
    "env": "CartPole-v1",
    "framework": "torch",
    "log_level": "INFO"
    },
    time_budget_s=100
)
