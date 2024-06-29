import tensorflow as tf
from collections import deque
import random
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from game import Rocket
class AsteroidGameEnv:
    def __init__(self, player):
        self.reset()

    def reset(self):
        self.ship_x = 0
        self.fuel = 0
        self.health = 0
        self.asteroids = []
        return self.get_state()

    def get_state(self):
        state = [self.ship_x, self.fuel, self.health]
        for i in range(10):
            if i < len(self.asteroids):
                state.extend(self.asteroids[i])
            else:
                state.extend([0, 0, 0, 0])
        return np.array(state)

    def step(self, action):
        if action == 1:  # move left
            self.ship_x -= 0.1
        elif action == 2:  # move right
            self.ship_x += 0.1
        elif action == 3:  # shoot (not implemented in this example)
            pass

        # Example of fuel consumption and health decrease (simplified)
        self.fuel -= 1
        self.health -= 0.1

        self.asteroids = [(x + vx, y + vy, vx, vy) for x, y, vx, vy in self.asteroids]

        if random.random() < 0.1 and len(self.asteroids) < 10:
            new_asteroid = (random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1))
            self.asteroids.append(new_asteroid)

        reward, done = self.check_collisions()
        return self.get_state(), reward, done, {}

    def check_collisions(self):
        done = False
        reward = 0
        for x, y, vx, vy in self.asteroids:
            if np.hypot(x - self.ship_x, y - self.ship_y) < 0.1:
                reward = -10
                done = True
                break
        return reward, done


env = AsteroidGameEnv()
player = Rocket()
state_size = 44
action_size = 4
agent = DQN()
batch_size = 32
episodes = 1000

for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(500):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print(f"episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2}")
            break
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)
    if e % 50 == 0:
        agent.save(f"asteroid-dqn-{e}.keras")
