from NoccaEnv import NoccaEnv
from Input import Input
from CPU.CPU import CPU
import os
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()


Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

steps_done = 0

BATCH_SIZE = 64
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

NUM_ACTION = 45  # 5*(8 + 1) (駒数*(周り8箇所+ゴール))


def select_action(state):
    # stateはtouchに変換済み
    global steps_done

    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(NUM_ACTION)]], device=device, dtype=torch.long)


def best_action(state, rank):
    # rank番目に価値が大きな行動を返す rank=0,1,2,...
    # stateを1次元に
    state = state.flatten()
    # stateをPytorch Tensorに
    state = torch.from_numpy(state.astype(np.float32)).clone()
    # Add a batch dimension (BCHW)
    state = state.unsqueeze(0).to(device)
    with torch.no_grad():
        # t.max(1) will return largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        output_np = policy_net(state).to('cpu').detach().numpy().copy()[0]
        return np.argsort(output_np)[-(rank + 1)]


episode_durations = []


def plot_durations():
    plt.figure(1)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def optimize_model():
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
    next_state_values[non_final_mask] = target_net(
        non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values,
                            expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


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


class DQN(nn.Module):
    def __init__(self, inputs, outputs):
        super(DQN, self).__init__()
        self.liner1 = nn.Linear(inputs, 180)
        self.liner2 = nn.Linear(180, 360)
        self.liner3 = nn.Linear(360, 180)
        self.liner4 = nn.Linear(180, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.liner1(x))
        x = F.relu(self.liner2(x))
        x = F.relu(self.liner3(x))
        x = self.liner4(x)

        return x


# Log
IsPrintLogs = True
WEIGHT_DIR = "./weight/"

nocca = NoccaEnv(1)
# myInputGenerator = Input(nocca)
cpuInputGenerator = CPU(player=-1, nocca=nocca, policy_type="Rule")
RLFirst = -1

# DQN network
policy_net = DQN(NoccaEnv.XRANGE * NoccaEnv.YRANGE *
                 NoccaEnv.ZRANGE, NUM_ACTION).to(device)
target_net = DQN(NoccaEnv.XRANGE * NoccaEnv.YRANGE *
                 NoccaEnv.ZRANGE, NUM_ACTION).to(device)
# ./weight/pre.pthがあればそれを読み込む
if os.path.isfile(WEIGHT_DIR + "pre.pth"):
    print("Load Pre-Trained Model")
    policy_net.load_state_dict(torch.load(WEIGHT_DIR + "pre.pth"))
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(1000)


# action_index = (動かす駒のindex[0-4])*8 + (周囲8箇所のindex[0-7])
# 各Pointのaction_indexを計算していき，引数と同じものを探す
# 動かせるactionかは確かめてない
def actionIndex_movePoint(action_index):
    pSurround = [
        [0, 1],
        [0, -1],
        [1, 0],
        [1, 1],
        [1, -1],
        [-1, 0],
        [-1, 1],
        [-1, -1],
        None  # 行き先ゴールのとき
    ]
    prevPoint = None
    nextPoint = None

    allPiecesPoints = nocca.getAllPiecesPoints()
    checkingActoinIndex = 0
    for p in allPiecesPoints:
        for pi in pSurround:
            if checkingActoinIndex == action_index:
                prevPoint = p
                if pi is None:
                    if nocca.isMyTurn == 1:
                        nextPoint = nocca.MyGoalPoint
                    elif nocca.isMyTurn == -1:
                        nextPoint = nocca.OppGoalPoint
                else:
                    nextPoint = p + pi
                return prevPoint, nextPoint
            checkingActoinIndex += 1


def make_action(action_index):
    global IsPrintLogs
    RL_WIN_REWARD = 10
    RL_LOSE_REWARD = -10
    CANNOT_MOVE_REWARD = -1
    OTHER_STEP_REWARD = 0

    # RLが動かした後，cpuも動かす
    reward = 0
    done = False
    if IsPrintLogs:
        print("turn:{}".format(nocca.isMyTurn))
    # RLが動かす
    if nocca.isMyTurn == RLFirst:
        prevPoint, nextPoint = actionIndex_movePoint(action_index)
        # 動かせない駒だったら負の報酬だけ与えてstateは変更しない
        canMovePointsFrom = nocca.canMovePointsFrom(prevPoint)
        canMove = False
        for pc in canMovePointsFrom:
            if np.all(nextPoint == pc):
                # 動かせるとき
                canMove = True
                nocca.move(prevPoint, nextPoint, True)
                done = nocca.isGameOver
                if IsPrintLogs:
                    print("RL move")
                if nocca.winner == RLFirst:
                    # RLの勝利
                    reward = RL_WIN_REWARD
                    done = True
                    if IsPrintLogs:
                        print("RL win")
                    return reward, done
                break
        if not canMove:
            # 動かせないとき
            reward = CANNOT_MOVE_REWARD
            done = False
            if IsPrintLogs:
                print("CANNOT move")
            return reward, done
    if nocca.isMyTurn == -1 * RLFirst:
        # cpuが動かす
        prevPoint, nextPoint = cpuInputGenerator.getIput()
        nocca.move(prevPoint, nextPoint, True)
        if IsPrintLogs:
            print("cpu move")
        # cpuの勝利
        if nocca.winner == -1 * RLFirst:
            reward = RL_LOSE_REWARD
            done = True
            if IsPrintLogs:
                print("cpu win")
            return reward, done

    # RLとCPUが1ずつ動かし，ゲーム続行
    reward = OTHER_STEP_REWARD
    done = False
    if IsPrintLogs:
        print("continue")
    return reward, done


def train():
    global IsPrintLogs
    IsPrintLogs = False

    global RLFirst
    num_episodes = 500
    RLFirst = -1
    for i_episode in range(num_episodes):
        # Initialize the environment
        nocca.initState()
        state = nocca.getState()
        # stateを1次元に
        state = state.flatten()
        # stateをPytorch Tensorに
        state = torch.from_numpy(state.astype(np.float32)).clone()
        # Add a batch dimension (BCHW)
        state = state.unsqueeze(0).to(device)

        if RLFirst == -1:
            # cpuが動かす
            if nocca.isMyTurn == -1 * RLFirst:
                prevPoint, nextPoint = cpuInputGenerator.getIput()
                nocca.move(prevPoint, nextPoint, True)

        for t in count():
            # Select and perform an action
            action = select_action(state)
            reward, done = make_action(action.item())
            if IsPrintLogs:
                print("reward:{}\n".format(reward))
            reward = torch.tensor([reward], device=device)

            # Observe new state
            if not done:
                next_state = nocca.getState()
                # stateを1次元に
                next_state = next_state.flatten()
                # stateをPytorch Tensorに
                next_state = torch.from_numpy(
                    next_state.astype(np.float32)).clone()
                # Add a batch dimension (BCHW)
                next_state = next_state.unsqueeze(0).to(device)
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize_model()
            if done:
                episode_durations.append(t + 1)
                print("{}th".format(i_episode))
                print("RLFirst:{}".format(RLFirst))
                print("Winner:{}\n".format(
                    "RL" if nocca.winner == RLFirst else "CPU"))
                # nocca.render()
                # plot_durations()
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    # モデルを保存
    if not os.path.exists(WEIGHT_DIR):
        os.makedirs(WEIGHT_DIR)
    torch.save(policy_net.state_dict(), WEIGHT_DIR + "weight.pth")
    print('Complete')
    nocca.render()
    plt.ioff()
    plt.show()


def test():
    BATTLE_NUM = 100
    rl_win_num = 0

    MyInputGenerator = CPU(player=-1, nocca=nocca, policy_type="Random")
    for i in range(BATTLE_NUM):
        nocca.initState()

        while nocca.winner == 0:
            prevPoint = None
            nextPoint = None
            if(nocca.isMyTurn == RLFirst):
                rank = 0
                prevPoint, nextPoint = actionIndex_movePoint(
                    best_action(nocca.getState(), rank))  # 価値最大
                canMove = False
                while not canMove:
                    for canP in nocca.canMovePointsFrom(prevPoint):
                        if np.all(nextPoint == canP):
                            canMove = True
                            break
                    if not canMove:
                        rank += 1
                        prevPoint, nextPoint = actionIndex_movePoint(
                            best_action(nocca.getState(), rank))

            elif(nocca.isMyTurn == -1 * RLFirst):
                prevPoint, nextPoint = MyInputGenerator.getIput()

            nocca.move(prevPoint, nextPoint, True)

        if nocca.winner == RLFirst:
            rl_win_num += 1
            print("{}/{} end".format(i+1, BATTLE_NUM))

    print("Complete")
    print("Win Rate:{}% {}/{}".format(100*rl_win_num /
                                      BATTLE_NUM, rl_win_num, BATTLE_NUM))


def main():
    train()
    test()


if __name__ == "__main__":
    main()
