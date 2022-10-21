import gym
import creversi.gym_reversi
from creversi import *

import argparse
import os
import datetime
import math
import random
import numpy as np
from collections import namedtuple, deque
from itertools import count
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='model.pt')
parser.add_argument('--resume')
parser.add_argument('--batchsize', type=int, default=256)
parser.add_argument('--num_episodes', type=int, default=160000)
parser.add_argument('--log')
parser.add_argument('--temperature', type=float, default=0.5)
parser.add_argument('--eg_param', type=int, default = 10000)

args = parser.parse_args()

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filename=args.log, level=logging.DEBUG)

env = gym.make('Reversi-v0').unwrapped

# if gpu is to be used
gpu_num = 2
device = torch.device(f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu")
# device = torch.device(f"cuda:{gpu_num}")


######################################################################
# Replay Memory

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'next_actions', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


######################################################################
# DQN

from network.cnn10 import DQN

def get_state(board):
    features = np.empty((1, 2, 8, 8), dtype=np.float32)
    board.piece_planes(features[0])
    state = torch.from_numpy(features[:1]).to(device)
    return state

######################################################################
# Training
TEMP = args.temperature
EG_PARAM = args.eg_param
BATCH_SIZE = args.batchsize
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 2000
OPTIMIZE_PER_EPISODES = 16
TARGET_UPDATE = 4

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters(), lr=1e-5)

if args.resume:
    print('resume {}'.format(args.resume))
    checkpoint = torch.load(args.resume)
    target_net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

memory = ReplayMemory(131072)

def epsilon_greedy(state, legal_moves):
    sample = random.random()
    # eps_threshold = EPS_END + (EPS_START - EPS_END) * \
    #     math.exp(-1. * episodes_done / EPS_DECAY)
    #EG_PARAM->大きくなるにつれ探索しやすくなる
    eps_threshold = 1/(episodes_done/EG_PARAM +1)
    #線形
    # eps_threshold = -(1/num_episodes)*episodes_done + 1

    if sample > eps_threshold:
        with torch.no_grad():
            q = policy_net(state)
            _, select = q[0, legal_moves].max(0)
    else:
        select = random.randrange(len(legal_moves))
    return select

#tadaoオリジナル
def softmax1(state, legal_moves):
    with torch.no_grad():
        q = policy_net(state)
        log_prob = q[0, legal_moves] / TEMP
        #確率に従って選択
        select = torch.distributions.categorical.Categorical(logits=log_prob).sample()
    return select
#改良（定義通りに）
def softmax2(state, legal_moves):
    with torch.no_grad():
        q = policy_net(state)
        qt = q[0, legal_moves] / TEMP
        qt_exp = [math.exp(qt_num) for qt_num in qt]
        qt_exp_sum = sum(qt_exp)
        logits = torch.tensor(qt_exp) / qt_exp_sum
        #確率に従って選択
        select = torch.distributions.categorical.Categorical(logits=logits).sample()
    return select

def yaki_softmax(state, legal_moves):
    max_epi = args.num_episodes
    #yaki_softmax_1
    # yaki_temp = (math.exp(episodes_done/10000) - 1) / (math.exp(max_epi/10000) - 1)
    #yaki_softmax_2
    yaki_temp = 1.-(EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * episodes_done / EPS_DECAY))

    with torch.no_grad():
        if yaki_temp==0:
            select = random.randrange(len(legal_moves))
        else:
            q = policy_net(state)
            log_prob = q[0, legal_moves] / TEMP
            #確率に従って選択
            select = torch.distributions.categorical.Categorical(logits=log_prob).sample()
    return select

def yaki_softmax3(state, legal_moves):
    yaki_temp = (EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * episodes_done / EPS_DECAY))
    #TEMP -> 値が小さくなるほど選択がQ値通りになる 値が大きいほど選択はランダム
    with torch.no_grad():
        if yaki_temp<=0:
            yaki_temp = 0.00001
        q = policy_net(state)
        qt = q[0, legal_moves] / yaki_temp
        qt_exp = [math.exp(qt_num) for qt_num in qt]
        qt_exp_sum = sum(qt_exp)
        logits = torch.tensor(qt_exp) / qt_exp_sum
        #確率に従って選択
        select = torch.distributions.categorical.Categorical(logits=logits).sample()
    return select


def egreedy_softmax(state, legal_moves):#legal_moves->合法手の記号
    with torch.no_grad():
        q = policy_net(state)
        q_tensor = q[0, legal_moves]
        max_q, max_select = q_tensor.max(0)#max_select->legal_movesの添え字
        select_list = []
        for n, q in enumerate(q_tensor):#最大q値と近いものを探す
            if max_q - q < 0.01:
                select_list.append(n)

        if len(select_list) == 1:#近いものがない->e-greedy法
            sample = random.random()
            # eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            #     math.exp(-1. * episodes_done / EPS_DECAY)
            eps_threshold = 1/(episodes_done/EG_PARAM +1)

            if sample > eps_threshold:
                select = max_select
            else:
                select = random.randrange(len(legal_moves))
        else:#近いものがある->softmax法
            #q値が近いもの同士でランダム
            # select = random.choice(select_list)
            #softmax
            log_prob = q_tensor / TEMP
            select = torch.distributions.categorical.Categorical(logits=log_prob).sample()
    return select

def select_action(state, board):

    legal_moves = list(board.legal_moves)

    select = epsilon_greedy(state, legal_moves)
    # select = softmax1(state, legal_moves)
    # select = softmax2(state, legal_moves)
    # select = yaki_softmax(state, legal_moves)
    # select = yaki_softmax3(state, legal_moves)
    # select = egreedy_softmax(state, legal_moves)

    return legal_moves[select], torch.tensor([[legal_moves[select]]], device=device, dtype=torch.long)


######################################################################
# Training loop

def optimize_model(loss_que):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # 合法手のみ
    non_final_next_actions_list = []
    for next_actions in batch.next_actions:
        if next_actions is not None:
            non_final_next_actions_list.append(next_actions + [next_actions[0]] * (30 - len(next_actions)))
    non_final_next_actions = torch.tensor(non_final_next_actions_list, device=device, dtype=torch.long)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    # 合法手のみの最大値
    target_q = target_net(non_final_next_states)
    # 相手番の価値のため反転する#+のままだと相手が有利になる手を学習してしまう
    next_state_values[non_final_mask] = -target_q.gather(1, non_final_next_actions).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = next_state_values * GAMMA + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    logging.info(f"{episodes_done}: loss = {loss.item()}")

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    #########################
    max_len = 1000
    learn_stop = False
    loss_que.append(loss.item())
    if len(loss_que) > max_len:
        loss_que.popleft()
    mean_loss = sum(loss_que) / len(loss_que)

    if mean_loss < 0.005 and len(loss_que)==max_len:
        learn_stop = True

    ##########################
    return learn_stop






######################################################################
# main training loop

num_episodes = args.num_episodes
episodes_done = 0
loss_que = deque()
learn_stop = False
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    state = get_state(env.board)
    
    for t in count():
        # Select and perform an action
        move, action = select_action(state, env.board) #move:creversi用、action:pytorch学習用
        next_board, reward, done, is_draw = env.step(move)

        reward = torch.tensor([reward], device=device)

        # Observe new state
        if not done:
            next_state = get_state(next_board)
            next_actions = list(next_board.legal_moves)
        else:
            next_state = None
            next_actions = None

        # Store the transition in memory
        memory.push(state, action, next_state, next_actions, reward)

        if done:
            break

        # Move to the next state
        state = next_state

    episodes_done += 1

    if i_episode % OPTIMIZE_PER_EPISODES == OPTIMIZE_PER_EPISODES - 1:
        # Perform several episodes of the optimization (on the target network)
        learn_stop = optimize_model(loss_que)

        # Update the target network, copying all weights and biases in DQN
        if i_episode // OPTIMIZE_PER_EPISODES % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
    if learn_stop:
        break
print('save {}'.format(args.model))
torch.save({'state_dict': target_net.state_dict(), 'optimizer': optimizer.state_dict()}, args.model)

print('Complete')
env.close()