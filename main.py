import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from agents.dqn_agent import DQNAgent
from utils.train_logger import TrainLogger

def main():
    train()

def train(
    env_name: str = "LunarLander-v3",
    num_episodes: int = 1000,
    max_timesteps: int = 1000,
    log_every: int = 10
):
    # create environment using gymnasium
    env = gym.make(env_name)

    # get dimensions for the state and action space
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # initialize agent
    agent = DQNAgent(state_dim, action_dim)

    # initialize logger for tracking rewards and success rates for episodes
    logger = TrainLogger(log_dir="results")

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        success = False

        for t in range (max_timesteps):
            # select action using epsilon greedy
            action = agent.select_action(state)

            # apply action to environment and collect transition
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # store transition in the replay buffer
            agent.store_transition(state, action, reward, next_state, done)

            # update q network using sampled experiences
            agent.train_step()

            state = next_state
            total_reward += reward

            if done:
                # check if episode is considered successful
                success = (info.get("success", False) or reward >= 200)
                break
                
        # log rewards and success info for the episode
        logger.log_episode(total_reward, success)

        # print average reward over recent episodes at logging intervals
        if (episode + 1) % log_every == 0:
            avg_reward = np.mean(logger.rewards[-log_every:])
            print(f"Episode {episode+1}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
    
    # save reward/success plots and final model
    logger.save_plots(label="dqn")
    torch.save(agent.q_network.state_dict(), "results/dqn_model.pth")
    env.close()

if __name__ == "__main__":
    main()