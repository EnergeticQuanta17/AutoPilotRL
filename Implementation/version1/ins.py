from RLAgentBuilder import RLAgent

# Create an instance of RLAgent
agent = RLAgent()

# Train the agent for 10 iterations, saving the model 10 times
# The model is saved every 100 timesteps during each iteration
agent.learn_and_save(timestep=100, iterations=1)

# Load the gym environment and render the output for 5 episodes
agent.load(no_of_episodes=5)