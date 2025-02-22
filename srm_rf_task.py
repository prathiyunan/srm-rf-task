import gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Define the neural network for approximating the Q-value function
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define the agent that interacts with the environment
class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  
        self.gamma = 0.95   
        self.epsilon = 1.0   
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32

        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    # Store experience in memory
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Choose an action using an epsilon-greedy policy
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        # Convert state to tensor and predict Q-values
        state_tensor = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return np.argmax(q_values.cpu().numpy())

    # Train the model using a random batch of experiences from memory
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).to(self.device)
            target = reward
            if not done:
                # Bellman equation: Q_target = r + gamma * max_a' Q(next_state, a')
                target = reward + self.gamma * torch.max(self.model(next_state_tensor)).item()
            # Compute current Q-value for the chosen action
            current_q = self.model(state_tensor)[action]
            target_tensor = torch.tensor(target).to(self.device)
            loss = self.loss_fn(current_q, target_tensor)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        # Decay the exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Main loop to train the agent in the Cart-Pole environment
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]  
    action_size = env.action_space.n
    agent = Agent(state_size, action_size)
    episodes = 500

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            # Uncomment below to render the environment (slows down training)
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            # Penalize if the episode ends early
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"Episode: {e+1}/{episodes}, Score: {time}, Epsilon: {agent.epsilon:.2f}")
                break
            # Train the agent with the experience of the current step
            agent.replay()
