from collections import deque
import random
import numpy as np
import torch
import csv
from gameAI import GameAI
from constants import *
from DQN import *

MAX_MEM = 4000
BATCH_SIZE = 32
learning_rate = 0.001


def normalise(state):
    mean = np.mean(state)
    std = np.std(state)
    return np.array([(state - mean) / std for state in state])


class Agent:
    def __init__(self):
        self.num_games = 0
        self.epsilon = 0.3
        self.epsilon_decay = 0.995
        self.memory = deque(maxlen=MAX_MEM)
        self.model = DQN(45, 4)
        self.trainer = QTrainer(self.model, learning_rate, 0.95)

    def get_state(self, game):
        """Retrieves the state of the game at that instance and returns a np array"""
        state = game.get_states()
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        """Helps in evaluation for deep learning purpose"""
        self.memory.append((state, action, reward, next_state, done))

    def train_long(self):
        mini_batch = self.memory
        if len(self.memory) > BATCH_SIZE:
            mini_batch = random.sample(self.memory, BATCH_SIZE)

        states, actions, rewards, next_states, dones = zip(*mini_batch)
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.uint8)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short(self, state, action, reward, next_state, done):
        """Performs a computation on the state using the model and checks with the next state"""
        self.trainer.train_step(state, action, reward, next_state, done)

    def load(self, model_path):
        """Helps the saved model be used on"""
        # Load the model state dictionary
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # Set the model to evaluation mode

    def csv_saver(self, game_date, game_time, elapsed_time, reason, score, accuracy, hits):
        """Records important data on a single game iteration for inferencing purpose"""
        file_exists = os.path.exists(DATA_FILE) == 1

        if file_exists:
            file_empty = os.path.getsize(DATA_FILE) == 0
        else:
            file_empty = True

        if not file_exists:
            with open(DATA_FILE, 'w', newline='') as file:
                writer = csv.writer(file)
                if file_empty:
                    writer.writerow(
                        ['Game Date', 'Game Time', 'Elapsed Time', 'Reason', 'Score', 'Accuracy', 'Asteroids Hit'])

        with open(DATA_FILE, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([game_date, game_time, elapsed_time, reason, score, accuracy,
                             hits])

    def get_action(self, state):
        """Returns actions for given state as list"""
        self.epsilon = 100 - self.num_games
        final_move = [0, 0, 0, 0]

        if np.random.rand() < self.epsilon/100:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state00 = state.clone().detach()
            pred = self.model(state00)
            move = torch.argmax(pred).item()
            final_move[move] = 1

        return final_move


def train():
    plot_score = []
    plot_mean = []
    record = -1
    frame_counter = 0
    last_10_scores = deque(maxlen=10)

    agent = Agent()
    game = GameAI()

    while True:
        first_state = agent.get_state(game)
        first_state = normalise(first_state)
        first_state = torch.tensor(first_state, dtype=torch.float)
        frame_counter += 1
        move = agent.get_action(first_state)
        if (frame_counter % SPAWN_RATE) == 0:
            game.spawn_asteroids()

        reward, done, score = game.play_action(move)
        print(f"Reward : {reward}, Done : {done}, Score: {score}")

        second_state = agent.get_state(game)
        agent.train_short(first_state, move, reward, second_state, done)
        agent.remember(first_state, move, reward, second_state, done)

        if done:
            agent.num_games += 1
            time_elap = round(game.end - game.start, 2)
            agent.csv_saver(game.gamedate, game.gametime, time_elap, game.death_reason,
                            game.game_score.get_score(), game.game_score.get_accuracy(), game.game_score.asteroids_hit)
            #agent.train_long()

            if score > record:
                print("Score when game ends : ", score)
                record = score
                agent.model.save_model()

            print(f"Game number : {agent.num_games}, Score : {score}, Record : {record}")
            plot_score.append(score)
            last_10_scores.append(score)
            mean_score = np.mean(last_10_scores)
            plot_mean.append(mean_score)
            game.reset_game()
            print(f"Mean score : {mean_score}")


if __name__ == '__main__':
    train()
