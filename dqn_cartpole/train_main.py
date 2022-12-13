#!/usr/bin/env python # -*- coding: utf-8 -*-

# @Time : 13.12.2022 14:53
# @Author : roman.held
# @Project: Deep_Q_Network.ipynb
# @File : train_main.py

import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from cartpole_agent import CartPoleAgent

import argparse
from tensorboardX import SummaryWriter

def createArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--learning_rate', default=5e-4)
    parser.add_argument('--replay_buffer_size', default=10000)
    parser.add_argument('--replay_buffer_coldrun', default=128)
    parser.add_argument('--epsilon_max', default=1)
    parser.add_argument('--epsilon_min', default=0.01)
    parser.add_argument('--epsilon_decay', default=0.994)
    parser.add_argument('--update_target_net', default=4)
    parser.add_argument('--n_episodes', default=2000)
    parser.add_argument('--max_t', default=1000)
    parser.add_argument('--gamma', default=0.99)
    parser.add_argument('--soft_update_tau', default=.5)


    args = parser.parse_args()
    return args

def createLogger():
    writer = SummaryWriter()
    return writer


def createEnv():
    env = gym.make('CartPole-v1')

    #env.seed(0)
    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)
    return env


def dqn(env, agent, logger, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    n = 0
    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        score = 0
        for t in range(max_t):
            n+=1
            action = agent.act(state, eps)
            next_state, reward, done, _, _ = env.step(action)
            loss = agent.step(state, action, reward, next_state, done)
            if loss:
                logger.add_scalar('train/loss', loss, n)
            state = next_state
            score += reward

            if done:
                logger.add_scalar('train/score', score, n)
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        logger.add_scalar('train/epsilon', eps, n)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 500.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores

if __name__ == '__main__':
    args = createArgs()
    logger = createLogger()
    env = createEnv()
    agent = CartPoleAgent(args, state_size=env.observation_space.shape[0], action_size=env.action_space.n, seed=0, device='cuda')
    scores = dqn(env, agent, logger, n_episodes=args.n_episodes, max_t=args.max_t, eps_start=args.epsilon_max, eps_end=args.epsilon_min, eps_decay=args.epsilon_decay)

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()