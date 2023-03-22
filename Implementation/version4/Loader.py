import os
from stable_baselines3 import PPO
import gym




class MegaLoader:
    def __init__(self, env_name, algo):
        self.env_name = env_name
        self.env = gym.make(self.env_name)
        self.env.reset()

        self.algorithm = algo
        pass

    def model_selector(self, all_models_dir):
        # print( os.listdir(all_models_dir)[-1])
        return os.listdir(all_models_dir)[-1]

        # return whole file name properly, with date and all
        pass

    def load(self, execution_number, no_of_episodes):
        # Input :- 
        #    1. execution_number - 
        #    2. no_of_episodes - 
        #    3. 

        # Returns :-
        #    Dictionary with the following items
            #    1. total_return       [int]
            #    2. return_per_episode [list]
            #    3.
            #    3.
            #    3.
            #    3.

        total_return = 0
        return_per_episode = []
        info_per_episode = []

        all_models_dir = f"../model/{self.env_name}/{self.algorithm}/{execution_number}"

        model_no = self.model_selector(all_models_dir)
        model_dir = f"{all_models_dir}/{model_no}"

        model = PPO.load(model_dir, env=self.env)

        for ep in range(no_of_episodes):
            obs = self.env.reset()
            done = False
            info = ""
            episode_return = 0
            
            while not done:
                action, _states = model.predict(obs)
                obs, rewards, done, info = self.env.step(action)
                episode_return += rewards

                # self.env.render()

            return_per_episode.append(episode_return)
            total_return += episode_return
            info_per_episode.append(info)

        model_test_info = {
            "total_return": total_return,
            "return_per_episode": return_per_episode,
            "info_per_episode": info_per_episode,
        }

        return model_test_info