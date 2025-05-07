from typing import Deque, Tuple, List
import random
import numpy as np 
from collections import deque 

class ReplayBuffer:
    def __init__(self, buffer_size: int, batch_size: int):
        # use a deque to store fixed size buffer of transitions
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        # append a new experience tuple to the buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # randomly sample a batch of transitions from buffer for training
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.uint8)
        )
    
    def __len__(self) -> int:
        # return current number of elements in buffer
        return len(self.buffer)

