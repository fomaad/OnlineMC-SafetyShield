#!/usr/bin/env python3
import os
import sys
import glob
import numpy as np

def main():
    if len(sys.argv) < 2:
        print("Usage: {} <directory>".format(sys.argv[0]))
        sys.exit(1)
    
    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print("Error: '{}' is not a valid directory.".format(directory))
        sys.exit(1)
    
    yaml_files = glob.glob(os.path.join(directory, "*.yaml"))
    if not yaml_files:
        print("No yaml files found in '{}'".format(directory))
        sys.exit(1)
    
    rewards = []
    for yaml_file in yaml_files:
        with open(yaml_file, "r") as f:
            for line in f:
                if "reward:" in line:
                    try:
                        reward_str = line.split("reward:")[1].strip()
                        reward = float(reward_str)
                        rewards.append(reward)
                    except Exception as e:
                        print("Could not parse reward in {}: {}".format(yaml_file, e))

    if not rewards:
        print("No rewards found in the .yaml files.")
        sys.exit(1)
    
    print("Rewards found:")
    for r in rewards:
        print(r)
    
    rewards_arr = np.array(rewards)
    avg = np.mean(rewards_arr)
    std = np.std(rewards_arr)
    print("\nAverage reward: {:.6f}".format(avg))
    print("Standard Deviation: {:.6f}".format(std))

if __name__ == "__main__":
    main()
