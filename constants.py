
MOVE_AHEAD = 'MoveAhead'
ROTATE_LEFT = 'RotateLeft'
ROTATE_RIGHT = 'RotateRight'
LOOK_UP = 'LookUp'
LOOK_DOWN = 'LookDown'
# DONE = 'Done'
TOMATO_DONE = 'Tomato_Done'
BOWL_DONE = 'Bowl_Done'
# add tomato done and bowl done
BASIC_ACTIONS = [MOVE_AHEAD, ROTATE_LEFT, ROTATE_RIGHT, LOOK_UP, LOOK_DOWN, TOMATO_DONE, BOWL_DONE]

GOAL_SUCCESS_REWARD = 5
STEP_PENALTY = -0.01
FAILED_ACTION_PENALTY = -0.02
