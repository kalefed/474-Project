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
            grid, ((2, 2), (2, 2), (0, 0)), mode="constant", constant_values=5
        )
        return padded[y : y + 5, x : x + 5]
    except:
        return np.zeros((5, 5, 3), dtype=np.uint8)


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

    reward = 0.0
    grid_size = 10
    agent_x = agent_pos % grid_size
    agent_y = agent_pos // grid_size
    agent_coord = (agent_y, agent_x)


    in_enemy_fov = False
    for enemy in enemies:
        # Check if the agent is in the enemy's field of view
        if agent_coord in enemy.get_fov_cells():
            in_enemy_fov = True
            break

    # Bravery-based exploration
    if new_cell_covered:
        if in_enemy_fov:
            reward += 1.5  # brave tile
        else:
            reward += 1.0  # normal tile
    else:
        reward -= 0.3 # been to this space already

    # Proximity bravery THIS MIGHT CAUSE ISSUE WITH KEEPING THE AGENT NEARBY ENEMY AND NOT UNCOVERING THE MAP
    if new_cell_covered:
        for enemy in enemies:
            dist = abs(agent_x - enemy.x) + abs(agent_y - enemy.y)
            if 1 <= dist <= 4:
                reward += 0.15  # flirting with danger

    if cells_remaining == 0:
        reward += 15.0 + 0.1 * steps_remaining # bonus finishing early 

    if game_over:
        reward -= 10.0

    return reward

    
