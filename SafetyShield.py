import gymnasium as gym
import maude
import math
from highway_env.road.lane import AbstractLane

TTC_SAFE_THRESHOLD = 2.0
# note that one addition depth is needed to specify which NPC will be considered
SEARCH_DEPTH = 4
ACTIONS_ALL = {
    0: 'LANE_LEFT',
    1: 'IDLE',
    2: 'LANE_RIGHT',
    3: 'FASTER',
    4: 'SLOWER'
}

def current_ttc(propMod, stateTerm):
    current_ttc = propMod.parseTerm(f"ttc({stateTerm})")
    current_ttc.reduce()
    ttcPairArgs = current_ttc.arguments()
    ttcT = next(ttcPairArgs)
    idT = next(ttcPairArgs)
    try:
        ttc = float(str(ttcT))
        id = int(str(idT))
        return ttc,id
    except ValueError:
        print(f"[Exception]: TTC computation failed ({current_ttc})")

def choose_action(env, actionList, state_info):
    discardedActions = []
    for action in actionList:
        if validate(env, action, state_info):
            return int(action), discardedActions
        discardedActions.append(int(action))

    # no action passed, then slow down
    print(f"[SafetyShield] Time {state_info['timeStamp']}, "
          f"all actions seem to be unsafe, take SLOWER.")
    return 4, discardedActions

def validate(env, action, state_info):
    # ignore SLOWER action
    if action == 4:
        print(f"[SafetyShield] Time {state_info['timeStamp']}, "
              f"action {ACTIONS_ALL[action]} ignored.")
        return True

    # when the action is unavailable
    # E.g., LEFTLANE when on the left-most lane
    if not(action in env.get_available_actions()):
        print(f"[SafetyShield] Time {state_info['timeStamp']}, "
              f"action {ACTIONS_ALL[action]} discarded due to unavailable.")
        return False

    propMod = maude.getModule("PROPOSITIONS")

    assign_cond = maude.AssignmentCondition(propMod.parseTerm('< T:Float ; ID:Int >'),
                                            propMod.parseTerm('ttc(S:Sys)'))
    ttc_cond = maude.EqualityCondition(propMod.parseTerm(f'T:Float < {TTC_SAFE_THRESHOLD}'),
                                       propMod.parseTerm('true'))

    s_var = propMod.parseTerm('S:Sys')
    ego_x = state_info['ego']['x']
    ego_y = state_info['ego']['y']
    ego_vx = state_info['ego']['vx']
    ego_vy = state_info['ego']['vy']
    ego_heading = state_info['ego']['heading']
    egoInitStr = f'veh(1000, vector2({ego_x}, {ego_y}), vector2({ego_vx}, {ego_vy}), {ego_heading}) # {action} # 1.0'

    npcsInitStr = ""
    for id, npc in enumerate(state_info['npcs']):
        npc_x = npc['x']
        npc_y = npc['y']
        npc_vx = npc['vx']
        npc_vy = npc['vy']
        npc_heading = npc['heading']

        if npcsInitStr != "":
            npcsInitStr += " & "
        npcsInitStr += f'veh({id}, vector2({npc_x}, {npc_y}), vector2({npc_vx}, {npc_vy}), {npc_heading})'

    stateStr = "-1 | " + egoInitStr + " | (" + npcsInitStr + ")"
    initT = propMod.parseTerm(stateStr)

    searchResult = initT.search(maude.ANY_STEPS, s_var, condition=[assign_cond, ttc_cond], depth=SEARCH_DEPTH + 1)
    firstResult = next(searchResult, None)

    if firstResult is None:
        print(f"[SafetyShield] Time {state_info['timeStamp']}, "
              f"action {ACTIONS_ALL[action]} passed.")
        return True
    else:
        print(f"[SafetyShield] Time {state_info['timeStamp']}, "
              f"action {ACTIONS_ALL[action]} discarded due to unsafe.")
        return False