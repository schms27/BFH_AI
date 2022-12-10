from collections import defaultdict
import gym
import sys
from numpy.lib.nanfunctions import nanargmax
import numpy as np
from tqdm import trange # Processing Bar


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def generate_episode(env, Q, epsilon, nActions):
  episode = []
  state, _ = env.reset()
  while True:
    probability = get_probability(Q[state], epsilon, nActions)
    action = np.random.choice(np.arange(nActions), p=probability) if state in Q else env.action_space.sample()

    next_state, reward, terminated, truncated, info = env.step(action)

    episode.append((state, action, reward))
    state = next_state
    if terminated or truncated: # TODO check if it ends when falling into the lake
      break
  return episode

def get_probability(Q_s, epsilon, nActions):
  policy_s = np.ones(nActions) * epsilon / nActions

  best_action = np.argmax(Q_s)
  policy_s[best_action] =1 - epsilon + (epsilon/ nActions)
  return policy_s


def update_Q_monte_carlo(env, episode, Q, alpha, gamma):
  states, actions, rewards = zip(*episode)
  discounts = np.array([gamma**i for i in range(len(rewards)+1)])

  for state_idx, state in enumerate(states):
    nr_steps_after_state = len(rewards[state_idx:])
    old_Q = Q[state][actions[state_idx]]
    Q[state][actions[state_idx]] = old_Q + alpha*(sum(rewards[state_idx:]*discounts[:nr_steps_after_state]) - old_Q)
  return Q


def monte_carlo_control(env, num_episodes, alpha, gamma=1.0, eps_start=1.0, eps_decay=.99999, eps_min=0.05):
    nActions = env.action_space.n
    Q = defaultdict(lambda: np.zeros(nActions))
    epsilon = eps_start
    t = trange(num_episodes)

    reward_episode = []

    for i_episode in t:
        #if i_episode % 1000 == 0:
        #    print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
        #    sys.stdout.flush()
        epsilon = max(epsilon*eps_decay, eps_min)
        episode = generate_episode(env, Q, epsilon, nActions)
        Q = update_Q_monte_carlo(env, episode, Q, alpha, gamma)

        # update progress bar
        _, _, rewards = zip(*episode)
        reward_episode.append(sum(rewards))
        t.set_description(f'Episode {i_episode + 1} MeanReward {round(sum(rewards)/len(rewards),4)} Epsilon {round(epsilon,3)}')
        t.refresh()
    policy = dict((k,np.argmax(v)) for k, v in Q.items())
    return policy, Q, reward_episode


def plot_game_board(map_def_string, ax):
    grid_shape = int(np.sqrt(len(map_def_string)))
    custom_map = np.array(list(map_def_string)).reshape(grid_shape, grid_shape)
    new_custom_map = np.full((grid_shape,grid_shape), 1)
    new_custom_map[custom_map == 'F'] = 1
    new_custom_map[custom_map == 'H'] = 0
    new_custom_map[custom_map == 'S'] = 3
    new_custom_map[custom_map == 'G'] = 2

    holes = np.argwhere(new_custom_map == 0 )

    print(custom_map)

    ax.cla()
    ax.imshow(new_custom_map, cmap="Set1")
    ax.grid(True)

    for hole in holes:
        ax.text(hole[1],hole[0], "Hole", ha='center', va='center')

    ax.text(0,0, "Start", ha='center', va='top')
    ax.text(grid_shape-1,grid_shape-1, "Goal", ha='center', va='center')
    return grid_shape


def save_map(Q, map_string, name="test.png"):
    """
    Plot the Q matrix and its policy for a given "4x4" frozen-lake game board
    """
    fig, ax = plt.subplots(figsize=(10,10))

    grid_shape = plot_game_board(map_string, ax)

    print(grid_shape)

    for key, values in Q.items():
            posx = key // grid_shape
            posy = key % grid_shape
            action = np.argmax(values)
            # print(f'position- x={posx} y={posy}---; key={key} --;values {values}, action = {action}, value = {values[action]}')
            # action = values.index(max_value)
            weight = 'normal'
            if action == 0: # Left
                dx=-1
                dy=0
                ax.text(posy, posx, "L: " + str(round(values[action], 3)), weight=weight,  ha='center', va='center')
            elif action == 1: # Down
                dx=0
                dy=1
                ax.text(posy, posx, "D: " + str(round(values[action], 3)), weight=weight,  ha='center', va='center')
            elif action == 2: # Right
                dx=1
                dy=0
                ax.text(posy, posx, "R: " + str(round(values[action], 3)), weight=weight,  ha='center', va='center')
            elif action == 3: # Up
                dx=0
                dy=-1
                ax.text(posy, posx, "U: " + str(round(values[action], 3)), weight=weight, ha='center', va='center')      
            arrow = mpatches.Arrow(y=posx,x=posy,dx=dx,dy=dy)
            ax.add_patch(arrow)
    fig.savefig("./"+name)




env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False)#, render_mode='human')
policy, Q, reward = monte_carlo_control(env, 500, 0.02)

print(f'policy: {policy}')
# plot the final Q:
#save_map(Q)
map_4x4 = 'SFFHFHFFFFHFHFFG'
map_8x8 = 'SFFFHFFFFFFHFHFFFFHFFFFFHFFFFFHFFFFFFFFHFFFFHFFHFFFFFHFFFFFHFFFG'
map_8x8 = 'SFFFFFFFFFFFFFFFFFFHFFFFFFFFFHFFFFFHFFFFFHHFFFHFFHFFHFHFFFFHFFFG'
save_map(Q, map_8x8)
