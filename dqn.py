import warnings
warnings.filterwarnings("ignore")  # Suppresses everything

import numpy as np
import random
from collections import deque
import torch
import pygame
import torch.nn as nn
import torch.optim as optim
from Flappy import FlappyBirdEnv  # Assuming your environment is in Flappy.py


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state)
        act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([t[0] for t in minibatch]))
        actions = torch.LongTensor(np.array([t[1] for t in minibatch]))
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch]))
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch]))
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch]))

        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_model(next_states).detach().max(1)[0]
        target = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(current_q.squeeze(), target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))
        self.update_target_model()

    def save(self, name):
        torch.save(self.model.state_dict(), name)


def preprocess_state(state):
    """Normalize state values for better training"""
    state = np.array(state)
    state[0] /= 512  # bird y position
    state[1] /= 10  # bird velocity (clipped)
    state[2] /= 288  # distance to pipe
    state[3] /= 512  # pipe top
    state[4] /= 512  # pipe bottom
    return state


def train_dqn():
    env = FlappyBirdEnv(render_mode=True)
    state_size = 5  # From get_state() in Flappy.py
    action_size = 2  # 0 = no action, 1 = flap
    agent = DQNAgent(state_size, action_size)
    episodes = 1000
    max_steps = 1000
    save_interval = 50

    for e in range(episodes):
        state = preprocess_state(env.reset())
        total_reward = 0

        for step in range(max_steps):
            for event in pygame.event.get():  # <-- ADD THIS CHECK
                if event.type == pygame.QUIT:
                    pygame.quit()  # Close Pygame
                    return
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = preprocess_state(next_state)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            agent.replay()

            if done:
                agent.update_target_model()
                break

        print(f"episode: {e}/{episodes}, score: {_['score']}, e: {agent.epsilon:.2f}, reward: {total_reward:.1f}")

        if e % save_interval == 0:
            agent.save(f"flappy_dqn_{e}.pth")

    agent.save("flappy_dqn_final.pth")


def test_dqn():
    env = FlappyBirdEnv(render_mode=True)
    state_size = 5
    action_size = 2
    agent = DQNAgent(state_size, action_size)
    agent.load("flappy_dqn_final.pth")
    agent.epsilon = 0.01  # Minimal exploration

    state = preprocess_state(env.reset())
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = preprocess_state(next_state)
        state = next_state
        total_reward += reward

    print(f"Test score: {_['score']}, Total reward: {total_reward}")


if __name__ == "__main__":
    train_dqn()
    # test_dqn()  # Uncomment to test after training