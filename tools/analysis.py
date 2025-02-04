import math
import os
import sys
import warnings

import TwoDimTTC
import pandas as pd
import yaml

# state_info is {'timeStamp': time_stamp, 'ego': ego, 'npcs': npcs}
def minTTC(state_info):
    ego = state_info['ego']
    npcs = state_info['npcs']
    ttc = 1e9
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
            for id, test in enumerate(trajectories['trajectories']):
                trajectory = test['test-' + str(id)]
                for state in trajectory:
                    ttc = minTTC(state)
                    if ttc < 1:
                        distr[0] += 1
                    elif ttc < 2:
                        distr[1] += 1
                    else:
                        distr[2] += 1

            s = sum(distr)
            return [i/s for i in distr]

        except yaml.YAMLError as exc:
            print(exc)

def ttcAnalyzeMulFile(directory):
    for file in os.listdir(directory):
        if file.endswith(".yaml"):
            tracefile = os.path.join(directory, file)
            print(f"{tracefile}: {ttcAnalyze(tracefile)}")

if __name__ == '__main__':
    # deault = "trajectories.yaml"
    # if len(sys.argv) > 1:
    #     deault = sys.argv[1]
    #     print(ttcAnalyze(deault))

    dir = sys.argv[1]
    ttcAnalyzeMulFile(dir)
