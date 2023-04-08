from MegaDHyperPilotRL import *

m = MegaD26()

m.learn_and_save(timestep=100, iterations=10)  ## Create a model after every 100 timsteps, for 10 times

m.load(no_of_episodes=5)