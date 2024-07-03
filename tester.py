from agent import Agent
from gameAI import GameAI
import numpy as np

agent = Agent()
env = GameAI()

agent.load("./model/model.pth")

# Evaluate the model
for e in range(10):
    state = env.reset_game()
    for time in range(500):
        action = agent.get_action(state)
        reward, done, score = env.play_action(action)
        if done:
            print(f"test episode: {e+1}/10, score: {time}")
            break
