from time import sleep
import gym
import numpy as np
from gym.envs.toy_text.frozen_lake import generate_random_map
from collections import defaultdict


# env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)

env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='human', desc=generate_random_map(size=4))

env.action_space.seed()
obs = env.reset()

print(f'the current space:')
print(f'obs. space:     {env.observation_space}')
print(f'action space:   {env.action_space}')

while True:
    reward = 0
    obs = env.reset()
    while True:
        action = env.action_space.sample()
        new_obs, reward, terminated, truncated, info = env.step(action)
        print(f'{obs} -- {reward} -- {terminated} -- {truncated} -- {info}')
        obs = new_obs
        if terminated or truncated: ##  break when falling off the cliff
            print(f'End of the game! Reward: {reward}')
            break

    if reward:
        print(reward)
        break


sleep(10)