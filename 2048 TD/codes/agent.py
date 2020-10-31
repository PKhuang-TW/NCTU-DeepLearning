from board import board
from action import action
from weight import weight
from array import array
from episode import episode
import random
import sys
import copy
import numpy as np

class agent:
    """ base agent """
    
    def __init__(self, options = ""):
        self.info = {}
        options = "name=unknown role=unknown " + options
        for option in options.split():
            data = option.split("=", 1) + [True]
            self.info[data[0]] = data[1]
        return
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        return
    
    def open_episode(self, flag = ""):
        return
    
    def close_episode(self, flag = ""):
        return
    
    def take_action(self, state):
        return action()
    
    def check_for_win(self, state):
        return False
    
    def property(self, key):
        return self.info[key] if key in self.info else None
    
    def notify(self, message):
        data = message.split("=", 1) + [True]
        self.info[data[0]] = data[1]
        return
    
    def name(self):
        return self.property("name")
    
    def role(self):
        return self.property("role")


class random_agent(agent):
    """ base agent for agents with random behavior """
    
    def __init__(self, options = ""):
        super().__init__(options)
        seed = self.property("seed")
        if seed is not None:
            random.seed(int(seed))
        return
    
    def choice(self, seq):
        target = random.choice(seq)
        return target
    
    def shuffle(self, seq):
        random.shuffle(seq)
        return

    def close_episode(self, ep, flag = ""):
        return 


class weight_agent(agent):
    """ base agent for agents with weight tables """
    
    def __init__(self, options = ""):
        super().__init__(options)
        
        self.episode = episode()
        self.net = []
        self.alpha = 0.025
        load = self.property("load_weight")
        init = self.property("init")
        alpha = self.property("alpha")
        if load is not None:
            self.load_weights(load)
        else:
            self.init_weights()
        if alpha is not None:
            self.alpha = alpha
        return
    
    def __exit__(self, exc_type, exc_value, traceback):
        save = self.property("save")
        if save is not None:
            self.save_weights(save)
        return
    
    def init_weights(self):
        self.net += [weight(16**6)] # feature for line [0 1 4 5 8 9] includes 16*16*16*16*16*16 possible
        self.net += [weight(16**6)] # feature for line [1 2 5 6 9 10] includes 16*16*16*16*16*16 possible
        self.net += [weight(16**6)] # feature for line [2 6 10 9 14 13] includes 16*16*16*16*16*16 possible
        self.net += [weight(16**6)] # feature for line [3 7 11 10 15 14] includes 16*16*16*16*16*16 possible
        return
    
    def load_weights(self, path):
        print('Loading Weight...')
        input = open(path, 'rb')
        size = array('L')
        size.fromfile(input, 1)
        size = size[0]
        for i in range(size):
            self.net += [weight()]
            self.net[-1].load(input)
        return
    
    def save_weights(self, path):
        output = open(path, 'wb')
        array('L', [len(self.net)]).tofile(output)
        for w in self.net:
            w.save(output)
        return

    def open_episode(self, flag = ""):
        self.episode.clear()
        return

    # Close episode and Update weight
    def close_episode(self, ep, flag = ""):
        episode = ep[2:].copy()
        episode.reverse()
        
        def best_action(state):  # Return the best action
            expValues = []
            rewards = []
            for op in range(4):
                tmpBoard = copy.copy(state)
                rewards.append(tmpBoard.slide(op))  # get the reward of afterstate
                if rewards[-1] == -1:
                    # When the action is not allowed (reward==-1),
                    # it is impossible to take the action
                    expValues.append(-float("inf"))
                else:
                    expValues.append(rewards[-1] + self.lineValue(tmpBoard))
            best_move = np.argmax(expValues)
            return best_move, rewards[best_move]
        
        for idx in range(1, len(episode), 2):
            if idx == 1:  # Update the last state as 0
                idx0, idx1, idx2, idx3 = self.lineIndex(episode[2][0])
                self.net[0][idx0] = 0
                self.net[1][idx1] = 0
                self.net[2][idx2] = 0
                self.net[3][idx3] = 0
                continue
            sPrime = copy.copy(episode[idx][0])  # State s'
            sPrime2 = copy.copy(episode[idx-1][0])  # State s''
            tmpBoard = copy.copy(sPrime2)
            actionNext, rewardNext = best_action(tmpBoard)  # best action and reward at State s''
            tmpBoard.slide(actionNext)
            sPrime2Next = copy.copy(tmpBoard)  # State s'(next)
            value = rewardNext + self.lineValue(sPrime2Next) - self.lineValue(sPrime)
            self.updateLineValue(board_state=sPrime, value=value)
        return
    
    # Update Weight
    def updateLineValue(self, board_state, value):
        idx0, idx1, idx2, idx3 = self.lineIndex(board_state)
        self.net[0][idx0] += (self.alpha) * (value)
        self.net[1][idx1] += (self.alpha) * (value)
        self.net[2][idx2] += (self.alpha) * (value)
        self.net[3][idx3] += (self.alpha) * (value)
        return

    # Get the index of feature
    def lineIndex(self, board_state):
        idx0 = 0
        idx1 = 0
        idx2 = 0
        idx3 = 0
        for i in range(3):
            for j in range(2):
                idx0 = 16*idx0 + board_state[4*i + j]
                idx1 = 16*idx1 + board_state[4*i + (j+1)]
        for i in range(4):
            idx2 = 16*idx2 + board_state[4*i + 2]
            idx3 = 16*idx3 + board_state[4*i + 3]
            if i>=2:
                idx2 = 16*idx2 + board_state[4*i + 1]
                idx3 = 16*idx3 + board_state[4*i + 2]
        return idx0, idx1, idx2, idx3


    # Get the expected value of state
    def lineValue(self, board_state):
        value = 0.0
        for i in range(8):
            board = copy.copy(board_state)
            if (i >= 4):
                board.transpose()
            board.rotate(i)
            idx0, idx1, idx2, idx3 = self.lineIndex(board)
            value += self.net[0][idx0] + self.net[1][idx1] + self.net[2][idx2] + self.net[3][idx3]
        return value



class learning_agent(agent):
    """ base agent for agents with a learning rate """
    
    def __init__(self, options = ""):
        super().__init__(options)
        self.alpha = 0.1
        alpha = self.property("alpha")
        if alpha is not None:
            self.alpha = float(alpha)
        return


class rndenv(random_agent):
    """
    random environment
    add a new random tile to an empty cell
    2-tile: 90%
    4-tile: 10%
    """
    
    def __init__(self, options = ""):
        super().__init__("name=random role=environment " + options)
        return
    
    def __str__(self):
        return 'Environment\'s Turn'
    
    def take_action(self, state):
        empty = [pos for pos, tile in enumerate(state.state) if not tile]
        if empty:
            pos = self.choice(empty)
            tile = self.choice([1] * 9 + [2])
            return False, action.place(pos, tile)
        else:
            return True, action()

    
class player(weight_agent):
    
    def __init__(self, options = ""):
        super().__init__("name=weight role=player " + options)
        return
    
    def __str__(self):
        return 'Player\'s Turn'
    
    def take_action(self, state):
        expValues = []
        rewards = []
        for op in range(4):
            tmpBoard = board(state)
            # get reward of afterstate
            rewards.append(tmpBoard.slide(op))
            if rewards[-1] == -1:
                # When the action is not allowed (reward==-1),
                # it is impossible to take the action
                expValues.append(-float("inf"))
            else:
                expValues.append(rewards[-1] + self.lineValue(tmpBoard))
        if max(rewards) == -1:
            # if all the reward==-1,
            # then gameover
            return True, action()
        best_move = np.argmax(expValues)
        return False, action.slide(best_move)