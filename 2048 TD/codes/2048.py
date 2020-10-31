import os
import time
from board import board
from action import action
from episode import episode
from statistic import statistic
from agent import player
from agent import rndenv
from tqdm import tqdm
import sys


# ㄓㄓ
if __name__ == '__main__':
    print('2048 Demo: ' + " ".join(sys.argv))
    print()

    path = '../weights/'
    folderName = time.strftime("%m%d_%M%S", time.localtime())
    folderPath = path + folderName

    if not os.path.isdir(folderPath):
        os.makedirs(folderPath)

    total, block, limit = 1000000, 1000, 1000
    play_args, evil_args = "", ""
    load_weight = None
    
    for para in sys.argv[1:]:
        if "--total=" in para:
            total = int(para[(para.index("=") + 1):])
        elif "--block=" in para:
            block = int(para[(para.index("=") + 1):])
        elif "--limit=" in para:
            limit = int(para[(para.index("=") + 1):])
        elif "--load_weight=" in para:
            load_weight = path + para[(para.index("=") + 1):]

    stat = statistic(total, block, limit)
    
    if load_weight:
        play_args += 'load_weight=' + load_weight

    with player(play_args) as play, rndenv(evil_args) as evil:
        for iteration, ep in enumerate(tqdm(range(stat.total))):
            play.open_episode("~:" + evil.name())
            evil.open_episode(play.name() + ":~")

            stat.open_episode(play.name() + ":" + evil.name())
            game = stat.back()

            idx = 1
            while True:
                who = game.take_turns(play, evil)
                gameOver, move = who.take_action(game.state())
                    
                if gameOver:
                    break
                else:
                    game.apply_action(move)
                idx += 1
            win = game.last_turns(play, evil)  # The winner
            stat.close_episode(win.name())
            play.close_episode(game.ep_moves, win.name())
            evil.close_episode(win.name())
            
            if (iteration+1)%1000 == 0:
                fileName = folderPath + '/{}.pt'.format(str(iteration+1))
                play.save_weights(fileName)