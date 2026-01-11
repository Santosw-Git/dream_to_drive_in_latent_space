
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

import os
import tensorboard

# Set the log directory
log_dir = "./ppo_logs"

# Launch TensorBoard
os.system(f"tensorboard --logdir {log_dir} --port 6006")



ENV_NAME = "CarRacing-v3"
TOTAL_TIMESTEPS = 10000
N_ENVS = 4
SAVE_DIR = "./ppo_models/"

import os
os.makedirs(SAVE_DIR, exist_ok=True)


env = make_vec_env(ENV_NAME, n_envs=N_ENVS)


model = PPO(
    "CnnPolicy",    # CNN policy because input is image
    env,
    verbose=1,
    tensorboard_log="./ppo_logs/",
    learning_rate=2.5e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
)


checkpoint_callback = CheckpointCallback(
    save_freq=100_00 // N_ENVS,
    save_path=SAVE_DIR,
    name_prefix="ppo_carracing"
)


model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)


model.save(os.path.join(SAVE_DIR, "ppo_carracing_final"))
print("✅ PPO training complete and model saved!")
