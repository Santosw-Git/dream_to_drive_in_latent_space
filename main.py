import gymnasium as gym
import numpy as np
import cv2
import os
from tqdm import trange

ENV_NAME = "CarRacing-v3"
SAVE_DIR = "world_model_dataset"
NUM_EPISODES = 100
MAX_STEPS = 1000

os.makedirs(SAVE_DIR, exist_ok=True)

RENDER = True  

if RENDER:
    env = gym.make(ENV_NAME, render_mode="human")
else:
    env = gym.make(ENV_NAME, render_mode="rgb_array")


all_obs = []
all_actions = []
all_rewards = []
all_dones = []

for ep in trange(NUM_EPISODES):
    obs, info = env.reset()
    
    for step in range(MAX_STEPS):
        action = env.action_space.sample()

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # terminated → failed or success condition
        # truncated → time limit reached
        print(obs.shape)
        obs_small = cv2.resize(obs, (64, 64))

        all_obs.append(obs_small)
        all_actions.append(action)
        all_rewards.append(reward)
        all_dones.append(done)

        obs = next_obs

        if done:
            break

env.close()

all_obs = np.array(all_obs, dtype=np.uint8)
all_actions = np.array(all_actions)
all_rewards = np.array(all_rewards)
all_dones = np.array(all_dones)

# Save
np.savez_compressed(
    os.path.join(SAVE_DIR, "dataset.npz"),
    observations=all_obs,
    actions=all_actions,
    rewards=all_rewards,
    dones=all_dones
)

print("Dataset saved!")
print("Observations shape:", all_obs.shape)


data = np.load("world_model_dataset/dataset.npz")["observations"]
# print(data["observations"].shape)
# print(data["actions"].shape)
# print(data["rewards"].shape)
# print(data["dones"])


import matplotlib
matplotlib.use("TkAgg")  # or "TkAgg" depending on your system
import matplotlib.pyplot as plt
import numpy as np

image_array = data[11]

plt.imshow(image_array)
plt.axis('off')  
plt.show()