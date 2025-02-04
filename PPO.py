import os

import gymnasium as gym
import maude
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import highway_env
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import AbstractLane
import utils

log_path = "highway_ppo"
model_path = os.path.join(log_path, "model")

env_config = {
    "action": {
        "type": "DiscreteMetaAction",
        "target_speeds": [0, 5, 10, 15, 20, 25, 30, 35, 40]},
    "lanes_count": 3,
    "collision_reward": -1,  # (-1)
    "right_lane_reward": 0.1,  # (0.1),
    "reward_speed_range": [0, 40],
    "high_speed_reward": 1,
}

def train():
    n_cpu = 8
    batch_size = 128

    # Create and configure the environment
    env = make_vec_env("highway-fast-v0", n_envs=n_cpu, vec_env_cls=SubprocVecEnv, env_kwargs={"config": env_config})

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
        n_steps=batch_size * 12 // n_cpu,
        batch_size=batch_size,
        n_epochs=10,
        learning_rate=5e-4,
        gamma=0.97,
        verbose=2,
        tensorboard_log=log_path,
    )

    # Train the agent
    model.learn(total_timesteps=int(400_000), progress_bar=True)
    model.save(model_path)

RECORD_TRAJECTORIES = True
SAFETY_SHIELD_ENABLE = True
trajectory_file_name = "trajectories.yaml"

if SAFETY_SHIELD_ENABLE:
    maude.init()
    maude.load("vehicle.maude")
    maude.load("fmodel.maude")

def test():
    # Load the trained model
    model = PPO.load(model_path)
    env = gym.make("highway-fast-v0", render_mode="human")

    env_config["observation"] = {
        "type": "Kinematics",
        "features": ["presence", "x", "y", "vx", "vy", "heading"],
    }
    env_config["simulation_frequency"] = 15
    env.configure(env_config)
    env.reset()

    env.training = False  # Disable normalization updates
    env.norm_reward = False

    filename = utils.get_filename_arg(trajectory_file_name)
    dx_range = AbstractEnv.PERCEPTION_DISTANCE
    dy_range = AbstractLane.DEFAULT_WIDTH * env.unwrapped.config["lanes_count"]
    ego_vehicle = env.unwrapped.vehicle
    speed_bound = ego_vehicle.MAX_SPEED - ego_vehicle.MIN_SPEED
    policy_frequency = env.unwrapped.config["policy_frequency"]

    test_runs = 200
    crashed_runs, total_reward, trajectories = utils.do_test(env, model, test_runs, True, SAFETY_SHIELD_ENABLE,
                                                             policy_frequency, dx_range, dy_range,speed_bound)

    print("\rCrashes:", len(crashed_runs), "/", test_runs, "runs",
          f"({len(crashed_runs) / test_runs * 100:0.1f} %)")

    if RECORD_TRAJECTORIES:
        utils.write_trajectories(trajectories, crashed_runs, total_reward, filename)

    env.close()


if __name__ == "__main__":
    to_train = False
    if to_train:
        train()

    test()