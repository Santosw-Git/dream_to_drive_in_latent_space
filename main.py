import gymnasium as gym
import numpy as np
import os
import cv2
import random
import pandas as pd


NUM_EPISODES = 1
DATASET_DIR = "dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

ACTIONS_FILE = os.path.join(DATASET_DIR, "actions.csv")
REWARDS_FILE = os.path.join(DATASET_DIR, "rewards.csv")


def mixed_policy(obs):
    """Returns action: [steer, throttle, brake]"""
    r = random.random()
    
    if r < 0.3:
        action = np.array([
            random.uniform(-1, 1),
            random.uniform(0, 1),
            random.uniform(0, 1)
        ])
    
    elif r < 0.6:
        action = np.array([0.0, 0.8, 0.0])
    
    else:
        gray = np.mean(obs, axis=2)
        left = np.mean(gray[:, :32])
        right = np.mean(gray[:, -32:])
        steer = np.clip((right - left) * 0.01, -1, 1)
        throttle = np.random.uniform(0.4, 0.8)
        brake = 0.0
        action = np.array([steer, throttle, brake])
    
    action += np.random.normal(0, 0.05, size=3)
    action[0] = np.clip(action[0], -1.0, 1.0)
    action[1:] = np.clip(action[1:], 0.0, 1.0)
    
    return action

env = gym.make("CarRacing-v3", render_mode="human")

frame_id = 0
actions_data = []
rewards_data = []


for episode in range(NUM_EPISODES):
    obs, info = env.reset()
    done = False
    steps = 0
    max_steps = random.randint(500, 1500)  
    
    while not done and steps < max_steps:
        action = mixed_policy(obs)
        
        obs_next, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        img_path = os.path.join(IMAGES_DIR, f"frame_{frame_id:06d}.png")
        cv2.imwrite(img_path, cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
        
        actions_data.append([frame_id, *action])
        
        rewards_data.append([frame_id, reward])
        
        obs = obs_next
        frame_id += 1
        steps += 1



actions_df = pd.DataFrame(actions_data, columns=["frame_id", "steer", "throttle", "brake"])
rewards_df = pd.DataFrame(rewards_data, columns=["frame_id", "reward"])

actions_df.to_csv(ACTIONS_FILE, index=False)
rewards_df.to_csv(REWARDS_FILE, index=False)

env.close()
print(f"Data collection finished! Total frames: {frame_id}")
