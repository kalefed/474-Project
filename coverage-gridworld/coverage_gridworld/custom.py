import numpy as np
from gymnasium.spaces import Box
import pprint

"""
Feel free to modify the functions below and experiment with different environment configurations.
"""


def observation_space(env):
    """Defines a 5x5 RGB observation space"""
    return Box(low=0, high=255, shape=(5, 5, 3), dtype=np.uint8)


def observation(grid):
    """Returns a 5x5 grid around the agent"""
    # 1. Always return something, even if broken
    if grid is None:
        return np.zeros((5, 5, 3), dtype=np.uint8)

    # 2. Simple 5x5 view (with edge padding)
    try:
        y, x = np.argwhere(np.all(grid == [160, 161, 161], axis=-1))[0]
        padded = np.pad(
            grid, ((2, 2), (2, 2), (0, 0)), mode="constant", constant_values=0
        )
        return padded[y : y + 5, x : x + 5]
    except:
        return np.zeros((5, 5, 3), dtype=np.uint8)


directions = {
    0: 1,
    1: 2,
    2: 3,
    3: 4,
}


def reward(info: dict) -> float:
    """
    Function to calculate the reward for the current step based on the state information.

    The info dictionary has the following keys:
    - enemies (list): list of `Enemy` objects. Each Enemy has the following attributes:
        - x (int): column index,
        - y (int): row index,
        - orientation (int): orientation of the agent (LEFT = 0, DOWN = 1, RIGHT = 2, UP = 3),
        - fov_cells (list): list of integer tuples indicating the coordinates of cells currently observed by the agent,
    - agent_pos (int): agent position considering the flattened grid (e.g. cell `(2, 3)` corresponds to position `23`),
    - total_covered_cells (int): how many cells have been covered by the agent so far,
    - cells_remaining (int): how many cells are left to be visited in the current map layout,
    - coverable_cells (int): how many cells can be covered in the current map layout,
    - steps_remaining (int): steps remaining in the episode.
    - new_cell_covered (bool): if a cell previously uncovered was covered on this step
    - game_over (bool) : if the game was terminated because the player was seen by an enemy or not
    """
    enemies = info["enemies"]
    agent_pos = info["agent_pos"]
    total_covered_cells = info["total_covered_cells"]
    cells_remaining = info["cells_remaining"]
    coverable_cells = info["coverable_cells"]
    steps_remaining = info["steps_remaining"]
    new_cell_covered = info["new_cell_covered"]
    game_over = info["game_over"]

    # IMPORTANT: You may design a reward function that uses just some of these values. Experiment with different
    # rewards and find out what works best for the algorithm you chose given the observation space you are using
    reward = 0

    # NOTE - Kales reward stuff (WIP)
    # if the agent is in the same row or column of future FOV
    future_fov = []

    for enemy in enemies:
        x = enemy.x
        y = enemy.y
        orientation = enemy.orientation

        next_direction = directions[orientation]
        next_fov = []

        for i in range(1, 5):  # the evil guy can see 4 blocks
            if next_direction == 0:  # LEFT
                fov_row, fov_col = y, x - i
            elif next_direction == 1:  # DOWN
                fov_row, fov_col = y + i, x
            elif next_direction == 2:  # RIGHT
                fov_row, fov_col = y, x + i
            else:  # UP
                fov_row, fov_col = y - i, x

            next_fov.append((fov_row, fov_col))

    if len(str(agent_pos)) == 1:
        unflattened_agent_pos = (0, agent_pos)
    else:
        unflattened_agent_pos = (int(str(agent_pos)[0]), int(str(agent_pos)[1]))

    if unflattened_agent_pos in next_fov:
        reward -= 1.0

    # NOTE - previous rewards
    # if cells_remaining/coverable_cells > 0.50:

    if new_cell_covered:
        reward += 1.0
    else:
        reward -= 0.75

    if game_over:
        if cells_remaining == 0:
            reward += 10.0
        else:
            reward -= 10.0

    return reward
