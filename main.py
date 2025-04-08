import numpy as np

size = 5
goal = (4, 4)
q_table = np.zeros((size, size, 4))
actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

def choose_action(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(4)
    return np.argmax(q_table[state[0], state[1]])

def update_q(state, action, reward, next_state, alpha, gamma):
    predict = q_table[state[0], state[1], action]
    target = reward + gamma * np.max(q_table[next_state[0], next_state[1]])
    q_table[state[0], state[1], action] += alpha * (target - predict)

for episode in range(500):
    state = (0, 0)
    total_reward = 0
    for step in range(100):
        action = choose_action(state, 0.1)
        next_state = (min(max(0, state[0] + actions[action][0]), size - 1),
                      min(max(0, state[1] + actions[action][1]), size - 1))
        reward = 1 if next_state == goal else -0.01
        update_q(state, action, reward, next_state, 0.1, 0.99)
        state = next_state
        total_reward += reward
        if state == goal:
            break
    if episode % 100 == 0:
        print(f"Episode {episode}, Total reward: {total_reward}")
