import os
import yaml
import numpy as np

def calculate_distances(directory):
    distances = []
    for file in os.listdir(directory):
        if file.endswith(".yaml"):
            file_path = os.path.join(directory, file)
            print(f"Processing {file_path}")
            with open(file_path) as stream:
                try:
                    trajectories = yaml.load(stream, Loader=yaml.Loader)
                    for id, test in enumerate(trajectories['trajectories']):
                        trajectory = test['test-' + str(id)]
                        initState = trajectory[0]
                        lastState = trajectory[-1]
                        distances.append(lastState['ego']['x'] - initState['ego']['x'])

                except yaml.YAMLError as exc:
                    print(exc)
    return distances

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Distance Analysis Tool')
    parser.add_argument('directory', type=str, help='directory containing trajectory YAML files')

    args = parser.parse_args()
    distances = np.array(calculate_distances(args.directory))
    print(f"num: {len(distances)}, mean: {np.mean(distances)}, std: {np.std(distances)}, max: {np.max(distances)}, min: {np.min(distances)}")
