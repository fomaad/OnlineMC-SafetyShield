import math
import sys
import warnings

import numpy as np
import yaml
import SafetyShield
import torch
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3 import PPO, DQN

FLOAT_ROUND_PREC = 3

def round_float(x):
    return round(float(x), FLOAT_ROUND_PREC)

def ego_absolute_kinematic(ego_vehicle, action=-1):
    y = -round_float(ego_vehicle.position[1])
    vy = -round_float(ego_vehicle.velocity[1])
    heading = round_float(ego_vehicle.heading * 180 / np.pi)
    if math.isclose(vy, 0.0):
        vy = 0.0
    if math.isclose(y, 0.0):
        y = 0.0
    if math.isclose(heading, 0.0):
        heading = 0.0

    return {
        'x': round_float(ego_vehicle.position[0]),
        'y': y,
        'vx': round_float(ego_vehicle.velocity[0]),
        'vy': vy,
        'heading': heading,
        'action': action
    }

# npc_observe is relative
# y and vy should be negated (making them increase from down to top)
def npc_absolute_kinematic(npc_observe, dx_bound, dy_bound, speed_bound, ego_kinematic):
    x = round_float(npc_observe[1] * dx_bound + ego_kinematic['x'])
    y = round_float(-npc_observe[2] * dy_bound + ego_kinematic['y'])
    vx = round_float(npc_observe[3] * speed_bound + ego_kinematic['vx'])
    vy = round_float(-npc_observe[4] * speed_bound + ego_kinematic['vy'])
    heading = round_float(npc_observe[5] * 180 / np.pi)
    if math.isclose(vy, 0.0):
        vy = 0.0
    if math.isclose(y, 0.0):
        y = 0.0
    if math.isclose(heading, 0.0):
        heading = 0.0

    return {
        'x': x,
        'y': y,
        'vx': vx,
        'vy': vy,
        'heading': heading,
    }

def dump_state(state, time_stamp, dx_bound, dy_bound, speed_bound, ego_vehicle, ego_action=-1):
    ego = ego_absolute_kinematic(ego_vehicle, ego_action)
    npcs = [npc_absolute_kinematic(state[i], dx_bound, dy_bound, speed_bound, ego)
                for i in range(1, state.shape[0])]
    return {'timeStamp': time_stamp, 'ego': ego, 'npcs': npcs}

def do_test(env, model, test_runs, reshape=False, shield_enable=False,
            policy_frequency=1, dx_range=200, dy_range=12, speed_bound=80):
    crashed_runs = []
    total_reward = 0
    trajectories = []

    for k in range(test_runs):
        print(f'\n Test run #{k}')
        state = env.reset()[0]
        done = False
        truncated = False
        step = 0
        test_reward = 0
        trajectory = []

        ego_vehicle = env.unwrapped.vehicle
        state_info = dump_state(state, step * policy_frequency, dx_range, dy_range, speed_bound, ego_vehicle)
        trajectory.append(state_info)

        while not done and not truncated:
            step += 1
            correct_state = state
            if reshape:
                correct_state = np.delete(state, -1, axis=1)

            obs_ts = correct_state.reshape((-1,) + model.observation_space.shape)
            obs_ts = obs_as_tensor(obs_ts, model.policy.device)
            with torch.no_grad():
                if isinstance(model, DQN):
                    action_probs = model.q_net(obs_ts).cpu().numpy().flatten()
                elif isinstance(model, PPO):
                    action_dist = model.policy.get_distribution(obs_ts)
                    action_probs = torch.softmax(action_dist.distribution.logits, dim=-1).cpu().numpy().flatten()
                else:
                    raise NotImplementedError

            ranked_actions = np.argsort(action_probs)[::-1]

            if shield_enable:
                new_action, discardedActions = SafetyShield.choose_action(env, ranked_actions, state_info)

                # write action discarded to current state before $action is applied
                if discardedActions != []:
                    trajectory[-1]['ego']['actions-discarded'] = discardedActions
            else:
                new_action = ranked_actions[0]

            # write action to state before $action is applied
            trajectory[-1]['ego']['action'] = int(new_action)

            # apply the action
            next_state, reward, done, truncated, info = env.step(new_action)
            state = next_state
            test_reward += float(reward)
            env.render()

            # write state after $action was applied
            ego_vehicle = env.unwrapped.vehicle
            state_info = dump_state(state, step * policy_frequency, dx_range, dy_range, speed_bound, ego_vehicle)
            trajectory.append(state_info)

            if info and info['crashed']:
                crashed_runs.append(k)

        total_reward += test_reward
        trajectories.append({
            f"test-{k}": {
                "trajectory": trajectory,
                "reward": test_reward,
            }
        })

        # debug
        if k % 20 == 0:
            print(f'Number of crashes: {len(crashed_runs)}')
    return crashed_runs, total_reward, trajectories

def write_trajectories(trajectories, crashed_runs, total_reward, filename):
    try:
        with open(filename, 'w') as yaml_file:
            yaml.dump({
                'trajectories': trajectories,
                'crashed-test': crashed_runs,
                'total-reward': total_reward,
            },
                yaml_file, default_flow_style=False, sort_keys=False)
    except IOError as e:
        print(f"Error saving output file: {e}")

def get_filename_arg(default):
    if len(sys.argv) < 2:
        warnings.warn(f"WARNING: File name for trajectories not provided. "
                      f"Used the default name {default}.")
        return default
    else:
        print(f"Trajectories will be saved to {sys.argv[1]}")
        return sys.argv[1]