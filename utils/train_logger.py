from typing import List
import matplotlib.pyplot as plt 
import os 
import numpy as np

class TrainLogger:
    def __init__(self, log_dir: str = "results"):
        # lists for storing rewards and success flags per episode
        self.rewards: List[float] = []
        self.success_flags: List[int] = []

        # create directory for logging output
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def log_episode(self, reward:float, success: bool):
        # save reward and success
        self.rewards.append(reward)
        self.success_flags.append(int(success))

    def save_plots(self, label: str = "dqn"):
        # plot rewards per episode
        episodes = list(range(1, len(self.rewards) + 1))

        # reward plot
        plt.plot(episodes, self.rewards, label="Reward")
        plt.title("Training Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.savefig(f"{self.log_dir}/{label}_reward_plot.png")
        plt.close()

        # return plot
        plt.plot(episodes, self.rewards, label="Return", color="orange")
        plt.title("Episodic Return")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.grid(True)
        plt.savefig(f"{self.log_dir}/{label}_return_plot.png")
        plt.close()

        # plot success rate (moving average)
        window = 50
        success_rate = np.convolve(self.success_flags, np.ones(window)/window, mode="valid")
        plt.plot(range(window, len(self.success_flags) + 1), success_rate, label="Success Rate")
        plt.title("Landing Success Rate")
        plt.xlabel("Episode")
        plt.ylabel("Success Rate")
        plt.grid(True)
        plt.savefig(f"{self.log_dir}/{label}_success_plot.png")
        plt.close()

    def average_last_n(self, n: int = 100):
        # return average metrics over last N episodes for evaluation
        return {
            "avg_reward": np.mean(self.rewards[-n:]),
            "success_rate": np.mean(self.success_flags[-n:]) * 100
        }
