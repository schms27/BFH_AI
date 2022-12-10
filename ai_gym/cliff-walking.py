import sys
from time import sleep
import gym
import numpy as np
from collections import defaultdict

from tqdm import trange


class CliffWalk:

    def __init__(self, alpha, gamma=1.0, eps_start=1.0, eps_decay=.99999, eps_min=0.05):
        self.cliffWalk = gym.make('CliffWalking-v0') #, render_mode='human')

        self.cliffWalk.action_space.seed(43)

        obs, info = self.cliffWalk.reset(seed=42)

        print(f'the current space:')
        print(f'obs. space:     {self.cliffWalk.observation_space}')
        print(f'action space:   {self.cliffWalk.action_space}')

        # Actions:
        # up = 0      # --
        # down = 1    # --
        # left = 2    # --
        # right = 3   # --

        self.epsilon = eps_start
        self.gamma = gamma
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.alpha = alpha

        # nActions: number of actions:
        self.nActions = self.cliffWalk.action_space.n
        self.Q = defaultdict(lambda: np.zeros(self.nActions))


    def walk_the_cliff(self):
        episode = []
        state, _ = self.cliffWalk.reset()
        while True:
            action = self.action_func(state) #cliffWalk.action_space.sample()
            new_obs, reward, terminated, truncated, info = self.cliffWalk.step(action)
            episode.append((state, action, reward))
            state = new_obs

            if terminated or truncated: # or reward <= -100: ##  break when falling off the cliff
                # print(f'End of the game! Reward: {reward}')
                break
        return episode


    def action_func(self, state):
        if state in self.Q:
            probability = self.get_policy(self.Q[state])
            return np.random.choice(np.arange(self.nActions), p=probability) 

        return self.cliffWalk.action_space.sample()


    def get_policy(self, Q_s): # epsilon-greedy
        best_action = np.argmax(Q_s) # this is the Q-Learing part and is off policy here
                                     # epsilon-greedy is policy and here he goes off policy
                                     # on policy would be random and/or if he has a choice for random
                                     # Here he is doing only greedy here
        policy_s =  np.ones(self.nActions) * self.epsilon / self.nActions
        policy_s[best_action] = 1 - self.epsilon + (self.epsilon / self.nActions)
        return policy_s


    def update(self,  episode): # policy improvement or control
        """ updates the action-value function estimate using the most recent episode """
        states, actions, rewards = zip(*episode)
        # prepare for discounting
        discounts = np.array([self.gamma**i for i in range(len(rewards)+1)])
        for i, state in enumerate(states):
            n_steps_after_state = len(rewards[i:])
            old_Q = self.Q[state][actions[i]]
            self.Q[state][actions[i]] = old_Q + self.alpha*(sum(rewards[i:]*discounts[:n_steps_after_state]) - old_Q) # calculation of the target/reward


    def mc_control(self, num_episodes, print_interval=1000):
        print('in mc_control')

        t = trange(num_episodes)
        for i_episode in t:
            #if i_episode % 1000 == 0:
            #    print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            #    sys.stdout.flush()
            self.epsilon = max(self.epsilon*self.eps_decay, self.eps_min)

            episode = self.walk_the_cliff()
            self.update(episode)
            # update progress bar
            _, _, rewards = zip(*episode)
            t.set_description(f'Episode {i_episode + 1} MeanReward {round(sum(rewards)/len(rewards),4)} Epsilon {round(self.epsilon,3)}')
            t.refresh()

        return dict((k,np.argmax(v)) for k,v in self.Q.items())


    def close(self):
        self.cliffWalk.close()


if __name__=="__main__":
    cliffWalker = CliffWalk(alpha=0.02)
    policy = cliffWalker.mc_control(10, print_interval=100)

    print(f'policy: {policy}')