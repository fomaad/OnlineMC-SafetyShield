import math
import os
import sys
import warnings

import TwoDimTTC
import pandas as pd
import yaml
import numpy as np

# state_info is {'timeStamp': time_stamp, 'ego': ego, 'npcs': npcs}
def minTTC(state_info):
    ego = state_info['ego']
    npcs = state_info['npcs']
    ttc = np.inf
    for npc in npcs:
        temp = fastTTC(ego,npc)
        if temp < ttc:
            ttc=temp
    return ttc

# ego and npc have the following form
# {
#   'x': ,
#   'y': ,
#   'vx': ,
#   'vy': ,
#   'heading':
# }
def fastTTC(ego, npc):
    sample_data = {
        'x_i': [ego['x']],  # Ego position
        'y_i': [ego['y']],
        'vx_i': [ego['vx']],  # Ego velocity
        'vy_i': [ego['vy']],
        'hx_i': [math.cos(ego['heading'])],  # Ego heading
        'hy_i': [math.sin(ego['heading'])],
        'length_i': 5,  # Ego length
        'width_i': 2,  # Ego width
        'x_j': [npc['x']],  # NPC position
        'y_j': [npc['y']],
        'vx_j': [npc['vx']],  # NPC velocity
        'vy_j': [npc['vy']],
        'hx_j': [math.cos(npc['heading'])],  # NPC heading
        'hy_j': [math.sin(npc['heading'])],
        'length_j': [5],  # NPC length
        'width_j': [2]  # NPC width
    }

    # Convert to a pandas DataFrame
    sample = pd.DataFrame(sample_data)
    return TwoDimTTC.TTC(sample).TTC[0]


def ttcAnalyze(trajectories_file):
    with open(trajectories_file) as stream:
        try:
            trajectories = yaml.load(stream, Loader=yaml.Loader)
            distr = [0,0,0]
            ttc_list = []
            for id, test in enumerate(trajectories['trajectories']):
                trajectory = test['test-' + str(id)]['trajectory']
                for state in trajectory:
                    ttc = minTTC(state)
                    if ttc < 1:
                        distr[0] += 1
                    elif ttc < 2:
                        distr[1] += 1
                    else:
                        distr[2] += 1
                    if ttc != np.inf:
                        ttc_list.append(ttc)

            s = sum(distr)
            return ([i/s for i in distr], len(ttc_list), np.min(ttc_list), np.max(ttc_list), np.mean(ttc_list), np.median(np.rint(ttc_list)), np.std(ttc_list))

        except yaml.YAMLError as exc:
            print(exc)

def ttcAnalyzeMulFile(directory, verbose=False):
    results = []
    for file in os.listdir(directory):
        if file.endswith(".yaml"):
            tracefile = os.path.join(directory, file)
            result = ttcAnalyze(tracefile)
            if verbose:
                print(f"{tracefile}: {result[0]} (num: {result[1]}, min: {result[2]}, max: {result[3]}, mean: {result[4]}, median: {result[5]}, std: {result[6]})")
            else:
                print(f"{tracefile}: {result[0]}")
            results.append(result[0])


    print(f"TTC < 1, 1 <= TTC < 2, TTC >= 2 dis: {np.mean(np.array(results), axis=0)}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='TTC Analysis Tool')
    parser.add_argument('directory', type=str, help='directory containing trajectory YAML files')
    parser.add_argument('-v', '--verbose', action='store_true', help='enable verbose output')

    args = parser.parse_args()
    ttcAnalyzeMulFile(args.directory, args.verbose)
