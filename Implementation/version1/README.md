# Building and Testing Custom RL Agents in OpenAI Gym

An implementation of a custom reinforcement learning agent built using the OpenAI Gym environment and Stable Baselines3 library. The agent can be trained using the Proximal Policy Optimization (PPO) algorithm with a Multi-Layer Perceptron (MLP) policy.



## Usage

### Running ins.py
Run the "ins.py" file to load, train and test the RL Agent with default hyperparameters.

```
python ins.py
```

This will automatically load the environment, set the hyperparameters for the agent, train the agent for 10 iterations with model saving every 100 timesteps, and test the agent's performance by running 5 episodes with the trained model.

_Note_: You can modify the hyperparameters in the "ins.py" file before running it, to customize the training and testing process for the agent.

## Detailed Usage

### RLAgent Class
This class is used as an user-friendly interface for building, training and testing custom RL Agents.

```
agent = RLAgent()
```
This will prompt the user to enter custom values for the environment, algorithm, and policy hyperparameters. 

### Training the Agent
```
agent.learn_and_save(timestep=100, iterations=10)
```

### Testing the Agent's Performance
```
agent.load(no_of_episodes=5)
```

This will prompt the user to select a trained model to use for testing, and will then run the agent for 5 episodes.