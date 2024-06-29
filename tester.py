# Load the model
agent.load("asteroid-dqn-950.keras")

# Evaluate the model
for e in range(10):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(500):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        state = np.reshape(next_state, [1, state_size])
        if done:
            print(f"test episode: {e+1}/10, score: {time}")
            break
