import os
import yaml
import numpy as np

def calculate(directory):
    distances = []
    speeds = []
    rewards = []
    for file in os.listdir(directory):
        if file.endswith(".yaml"):
            file_path = os.path.join(directory, file)
            print(f"Processing {file_path}")
            with open(file_path) as stream:
                try:
                    trajectories = yaml.load(stream, Loader=yaml.Loader)
                    for id, test in enumerate(trajectories['trajectories']):
                        testdata = test['test-' + str(id)]
                        trajectory = testdata['trajectory']
                        reward = testdata['reward']
                        initState = trajectory[0]
                        lastState = trajectory[-1]
                        distance = lastState['ego']['x'] - initState['ego']['x']
                        distances.append(distance)
                        speeds.append(distance / lastState['timeStamp'])
                        rewards.append(reward)

                except yaml.YAMLError as exc:
                    print(exc)
    return (distances, speeds, rewards)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Distance, Speed, and Reward Analysis Tool')
    parser.add_argument('directory', type=str, help='directory containing trajectory YAML files')

    args = parser.parse_args()
    (distances, speeds, rewards) = map(np.array,calculate(args.directory))

    print("Distances:")
    print(f"num: {len(distances)}, mean: {np.mean(distances)}, std: {np.std(distances)}, max: {np.max(distances)}, min: {np.min(distances)}")
    print("Speeds:")
    print(f"num: {len(speeds)}, mean: {np.mean(speeds)}, std: {np.std(speeds)}, max: {np.max(speeds)}, min: {np.min(speeds)}")
    print("Rewards:")
    print(f"num: {len(rewards)}, mean: {np.mean(rewards)}, std: {np.std(rewards)}, max: {np.max(rewards)}, min: {np.min(rewards)}")
