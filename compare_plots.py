import numpy as np
import matplotlib.pyplot as plt

def load_and_plot():
    labels = ["dqn", "double_dqn"]
    colors = ["blue", "green"]
    window = 50

    plt.figure(figsize=(10, 6))

    # Plot Reward
    for label, color in zip(labels, colors):
        rewards = np.load(f"results/{label}_rewards.npy")
        plt.plot(rewards, label=f"{label.upper()} Reward", color=color)

    plt.title("Reward Comparison")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/comparison_reward_plot.png")
    plt.close()

    # Plot Success Rate
    plt.figure(figsize=(10, 6))
    for label, color in zip(labels, colors):
        success = np.load(f"results/{label}_success.npy")
        smoothed = np.convolve(success, np.ones(window)/window, mode='valid')
        plt.plot(range(window, len(success)+1), smoothed, label=f"{label.upper()} Success", color=color)

    plt.title("Success Rate Comparison")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate (Moving Avg)")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/comparison_success_plot.png")
    plt.close()

if __name__ == "__main__":
    load_and_plot()
