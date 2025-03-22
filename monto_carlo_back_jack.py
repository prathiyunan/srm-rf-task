import gymnasium as gym
import numpy as np
import random
from collections import defaultdict

# Initialize Blackjack environment
env = gym.make('Blackjack-v1', sab=True)

class MonteCarloAgent:
    def __init__(self, actions, epsilon=1.0, epsilon_decay=0.9999):
        self.q_table = defaultdict(lambda: np.zeros(len(actions)))  # Q-values
        self.returns = defaultdict(list)  # Stores returns for averaging
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate
        self.actions = actions  # 0: Stick, 1: Hit

    def choose_action(self, state):
        """Epsilon-greedy policy for action selection."""
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)  # Explore
        return np.argmax(self.q_table[state])  # Exploit

    def update_q_table(self, episode):
        """Monte Carlo update: Compute returns and update Q-table."""
        G = 0  # Total return (sum of rewards)
        visited_states = set()

        # Iterate through episode in reverse (backward pass)
        for state, action, reward in reversed(episode):
            G += reward  # Accumulate reward
            if (state, action) not in visited_states:  # First-visit MC
                self.returns[(state, action)].append(G)
                self.q_table[state][action] = np.mean(self.returns[(state, action)])
                visited_states.add((state, action))

        self.epsilon *= self.epsilon_decay  # Reduce exploration

def train_blackjack(agent, env, episodes=100000):
    """Train agent using Monte Carlo method."""
    for episode_num in range(episodes):
        state, _ = env.reset()
        episode = []
        done = False

        # Generate episode
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state

        agent.update_q_table(episode)  # Update Q-values

        if episode_num % 10000 == 0:
            print(f"Episode {episode_num}/{episodes}, Epsilon: {agent.epsilon:.4f}")

    print("Training completed!")

def evaluate_blackjack(agent, env, episodes=10000):
    """Evaluate trained agent's win rate."""
    wins, losses, draws = 0, 0, 0

    for _ in range(episodes):
        state, _ = env.reset()
        done = False

        while not done:
            action = np.argmax(agent.q_table[state])  # Always exploit policy
            state, reward, done, _, _ = env.step(action)

        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1

    print(f"Results after {episodes} games:")
    print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")
    print(f"Win Rate: {wins / episodes:.2%}")

# Create agent and train it
actions = [0, 1]  # 0: Stick, 1: Hit
agent = MonteCarloAgent(actions)
train_blackjack(agent, env)

# Evaluate trained agent
evaluate_blackjack(agent, env)
