import tensorflow as tf
from collections import deque
import random
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from gameAI import GameAI

def train():
    plot_score = []
    plot_mean = []
    record = -1

    last_10_scores = deque(maxlen=10)

    agent = Agent()
    game = GameAI()

    while True:
        first_state = agent.get_state(game)
        move = agent.get_move(game, state)
        reward, done, score = game.play_step(move)

        second_state = agent.get_state(game)
        agent.train_short(first_state, move, reward, second_state, done)
        agent.remember(first_state, move, reward, second_state, done)

        if done:
            game.reset_game()
            agent.num_games += 1
            agent.train_long()

            if score > record:
                record = score
                agent.model.save_model()

            print(f"Game number : {agent.num_games}, Score : {score}, Record : {record}")
            plot_score.append(score)
            last_10_scores.append(score)
            mean_score = np.mean(last_10_scores)
            plot_mean.append(mean_score)


if __name__ == '__main__':
    train()