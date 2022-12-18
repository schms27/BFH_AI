#!/usr/bin/env python # -*- coding: utf-8 -*-

# @Time : 13.12.2022 14:53
# @Author : roman.held
# @Project: Deep_Q_Network.ipynb
# @File : train_main.py
import os
import sys
import gym
import random
import torch
import optuna
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from cartpole_agent import CartPoleAgent

import argparse
from tensorboardX import SummaryWriter

def createArgs(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.000565)
    parser.add_argument('--replay_buffer_size', type=int, default=18308)
    parser.add_argument('--replay_buffer_coldrun', type=int, default=830)
    parser.add_argument('--epsilon_max', type=float, default=0.596)
    parser.add_argument('--epsilon_min', type=float, default=0.0575)
    parser.add_argument('--epsilon_decay', type=float, default=0.9955)
    parser.add_argument('--update_target_net', type=int, default=1)
    parser.add_argument('--n_episodes', type=int, default=500)
    parser.add_argument('--max_t', type=int, default=10000)
    parser.add_argument('--gamma', type=float, default=0.9955)
    parser.add_argument('--soft_update_tau', type=float, default=0.2637)
    parser.add_argument('--show_plot', type=bool, default=True)
    parser.add_argument('--experiment_name', type=str, default="experiment")
    parser.add_argument('--seed', type=int, default=1337)


    args = parser.parse_args(args)
    return args

def createLogger():
    writer = SummaryWriter()
    return writer


def createEnv(seed):
    env = gym.make('CartPole-v1')
    if seed > 0:
        env.observation_space.seed(seed)
        env.action_space.seed(seed)

    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)
    return env

def plotScores(scores, show_plot, experiment_name, plt_subtitle="Scores"):
    savedir = 'plots/'
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    fig = plt.figure(figsize=[14, 8])
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title(experiment_name, fontsize=16, loc='right')
    plt.title(plt_subtitle, fontsize=8, loc='left')
    plt.savefig(f'plots/scores_{experiment_name}.png', bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close()


def dqn(env, agent, logger, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, trial=None):
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
        if trial and len(scores_window) > 0:
            trial.report(np.mean(scores_window), i_episode)
            if trial.should_prune():
                raise optuna.TrialPruned()
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
    return scores, i_episode

def main(args, trial=None):
    args = createArgs(args)
    logger = createLogger()
    env = createEnv(args.seed)
    agent = CartPoleAgent(args, state_size=env.observation_space.shape[0], action_size=env.action_space.n)
    scores, solved_in_i_episodes = dqn(env, agent, logger, n_episodes=args.n_episodes, max_t=args.max_t, eps_start=args.epsilon_max, eps_end=args.epsilon_min, eps_decay=args.epsilon_decay, trial=trial)

    plotScores(
        scores, 
        args.show_plot, 
        args.experiment_name, #f"Params: {args}"
        'Params:\n[batch:{bs}, lr:{lr:.{digits}f}, rpb_size:{rpb_size}, rpb_coldrun:{rpb_coldrun}\neps_min:{eps_min:.{digits}f}, eps_max:{eps_max:.{digits}f}, eps_decay:{eps_decay:.{digits}f}, update_target_net:{update_target_net}\nn_episodes:{n_episodes}, max_t:{max_t}, gamma:{gamma:1.{digits}f}, soft_update_tau:{soft_update_tau:.{digits}f}]'
            .format(bs=args.batch_size, lr=args.learning_rate, rpb_size=args.replay_buffer_size, rpb_coldrun=args.replay_buffer_coldrun, eps_min=args.epsilon_min, eps_max=args.epsilon_max, eps_decay=args.epsilon_decay, update_target_net=args.update_target_net, n_episodes=args.n_episodes, max_t=args.max_t, gamma=args.gamma, soft_update_tau=args.soft_update_tau, digits=4)
    )

    return scores, solved_in_i_episodes


if __name__ == '__main__':
    main(sys.argv[1:])