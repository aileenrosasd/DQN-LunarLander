import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from agents.dqn_agent import DQNAgent
from utils.train_logger import TrainLogger

def train(agent_label: str, use_double_dqn: bool, num_episodes: int = 1000):

    # create environment using gymnasium
    env = gym.make("LunarLander-v3")

    # get dimensions for the state and action space
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # initialize the DQN agent (vanilla or double based on flag)
    agent = DQNAgent(state_dim, action_dim, use_double_dqn=use_double_dqn)
    logger = TrainLogger(log_dir="results") # logger to track performance

    for episode in range(num_episodes):
        state, _ = env.reset()  # reset env and get initial state
        total_reward = 0        # total reward for the episode
        success = False         # success flag based on reward threshold

        for _ in range(1000):
            action = agent.select_action(state) # choose action
            next_state, reward, terminated, truncated, _ = env.step(action) # take action
            done = terminated or truncated  # check if episode has ended

            # store the experience in the replay buffer
            agent.store_transition(state, action, reward, next_state, done) 

            # perform training step
            agent.train_step()

            # update state and accumulate reward
            state = next_state
            total_reward += reward

            if done:
                # episode is considered a success if reward exceeds threshold
                success = reward >= 100
                break

        # log performance
        logger.log_episode(total_reward, success)

        # print every 10 eps
        if (episode + 1) % 10 == 0:
            avg = np.mean(logger.rewards[-10:])
            print(f"{agent_label.upper()} Episode {episode+1}, Avg Reward: {avg:.2f}, Epsilon: {agent.epsilon:.2f}")

    # save logs and trained model
    logger.save_data(label=agent_label)
    torch.save(agent.q_network.state_dict(), f"results/{agent_label}_model.pth")
    env.close()

def plot_comparisons():
    labels = ["dqn", "double_dqn"]
    colors = ["blue", "green"]
    window = 50 # window size for moving average smoothing

    # plots

    # Rewards
    plt.figure()
    for label, color in zip(labels, colors):
        rewards = np.load(f"results/{label}_rewards.npy")
        plt.plot(rewards, label=f"{label.upper()} Reward", color=color)
    plt.title("Episodic Reward Comparison")
    plt.xlabel("Episode #")
    plt.ylabel("Episodic Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/comparison_reward_plot.png")
    plt.close()

    # Return (same data as reward, styled differently)
    plt.figure()
    for label, color in zip(labels, colors):
        rewards = np.load(f"results/{label}_rewards.npy")
        plt.plot(rewards, label=f"{label.upper()} Return", linestyle="--", color=color)
    plt.title("Episodic Return Comparison")
    plt.xlabel("Episode #")
    plt.ylabel("Episodic Return")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/comparison_return_plot.png")
    plt.close()

    # Success Rate
    plt.figure()
    for label, color in zip(labels, colors):
        success = np.load(f"results/{label}_success.npy")
        smoothed = np.convolve(success, np.ones(window)/window, mode='valid')
        plt.plot(range(window, len(success)+1), smoothed, label=f"{label.upper()} Success", color=color)
    plt.title("Success Rate Comparison")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/comparison_success_plot.png")
    plt.close()

# metrics summary table
def print_metrics_summary():
    labels = ["dqn", "double_dqn"]
    print("\n=== Metrics Summary Table (Last 100 Episodes) ===")
    print(f"{'Metric':<25} {'DQN (Vanilla)':<20}{'DQN + Extension':<20}")

    metrics = {}
    for label in labels:
        rewards = np.load(f"results/{label}_rewards.npy")
        success = np.load(f"results/{label}_success.npy")
        avg_reward = np.mean(rewards[-100:])
        avg_return = np.mean(rewards[-100:])  # identical to episodic reward
        success_rate = np.mean(success[-100:]) * 100
        metrics[label] = (avg_reward, avg_return, success_rate)

    print(f"{'Average Episodic Reward':<25} {metrics['dqn'][0]:<20.2f} {metrics['double_dqn'][0]:<20.2f}")
    print(f"{'Average Return':<25} {metrics['dqn'][1]:<20.2f} {metrics['double_dqn'][1]:<20.2f}")
    print(f"{'Success Rate (%)':<25} {metrics['dqn'][2]:<20.2f} {metrics['double_dqn'][2]:<20.2f}")

        

def main():
    print("Training Vanilla DQN...")
    train(agent_label="dqn", use_double_dqn=False)

    print("\nTraining Double DQN...")
    train(agent_label="double_dqn", use_double_dqn=True)

    # plots
    plot_comparisons()

    # metrics
    print_metrics_summary()
    


if __name__ == "__main__":
    main()