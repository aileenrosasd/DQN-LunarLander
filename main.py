import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from agents.dqn_agent import DQNAgent

def main():
    train()

def train(
    env_name: str = "LunarLander-v3",
    num_episodes: int = 1000,
    max_timesteps: int = 1000,
    log_every: int = 10
):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim, action_dim)
    rewards_per_episode = []
    success_flags = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        success = False

        for t in range (max_timesteps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            total_reward += reward

            if done:
                success = info.get("success", False) or reward >= 200
                break

        rewards_per_episode.append(total_reward)
        success_flags.append(int(success))

        if (episode + 1) % log_every == 0:
            avg_reward = np.mean(rewards_per_episode[-log_every:])
            print(f"Episode {episode+1}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    env.close()

    plt.plot(rewards_per_episode)
    plt.title("DQN Training Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig("results/dqn_reward_plot.png")
    plt.close()

    torch.save(agent.q_network.state_dict(), "results/dqn_model.pth")


if __name__ == "__main__":
    main()