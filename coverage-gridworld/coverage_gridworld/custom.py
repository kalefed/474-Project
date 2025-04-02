import numpy as np
import gymnasium as gym

"""
Feel free to modify the functions below and experiment with different environment configurations.
"""


def observation_space(env: gym.Env) -> gym.spaces.Space:
    """Defines a 5x5 RGB observation space"""
    # OBS STRATEGY 2
    return gym.spaces.Box(low=0, high=255, shape=(env.num_cells,2), dtype=np.uint8)
    
    #OBS STRATEGY 1
    #return gym.spaces.Box(low=0, high=255, shape=(5, 5, 3), dtype=np.uint8)

def observation(grid: np.ndarray):
    """Returns a 5x5 grid around the agent"""
    # OBS STRATEGY 2
    agent = np.argwhere(np.all(grid == [160, 161, 161], axis=-1))[0]
    agent = np.array([agent])
    rcs = np.argwhere(np.all(grid == [0, 0, 0], axis=-1))
    obs = np.append(agent, rcs, axis=0)
    while len(obs) < 100:
        obs = np.append(obs, [[255,255]], axis=0)
    return np.array(obs)

    # OBS STRATEGY 1
    # y, x = np.argwhere(np.all(grid == [160, 161, 161], axis=-1))[0]
    # padded = np.pad(
    #     grid, ((2, 2), (2, 2), (0, 0)), mode="constant", constant_values=0
    # )
    # return padded[y : y + 5, x : x + 5]


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
    # reward = 0

    # # NOTE - Kales reward stuff (WIP)
    # # if the agent is in the same row or column of future FOV
    # future_fov = []

    # for enemy in enemies:
    #     x = enemy.x
    #     y = enemy.y
    #     orientation = enemy.orientation

    #     next_direction = directions[orientation]
    #     next_fov = []

    #     for i in range(1, 5):  # the evil guy can see 4 blocks
    #         if next_direction == 0:  # LEFT
    #             fov_row, fov_col = y, x - i
    #         elif next_direction == 1:  # DOWN
    #             fov_row, fov_col = y + i, x
    #         elif next_direction == 2:  # RIGHT
    #             fov_row, fov_col = y, x + i
    #         else:  # UP
    #             fov_row, fov_col = y - i, x

    #         next_fov.append((fov_row, fov_col))

    # if len(str(agent_pos)) == 1:
    #     unflattened_agent_pos = (0, agent_pos)
    # else:
    #     unflattened_agent_pos = (int(str(agent_pos)[0]), int(str(agent_pos)[1]))
    # if len(enemies) > 0:
    #     if unflattened_agent_pos in next_fov:
    #         reward -= 1.0

    # # NOTE - previous rewards
    # 

    # if new_cell_covered:
    #     reward += 1.0
    # else:
    #     reward -= 0.75

    # if game_over:
    #     if cells_remaining == 0:
    #         reward += 10.0
    #     else:
    #         reward -= 10.0

    # return reward


    #_______________________________ MILESTONE REWARD
    reward = 0.0

    # Base reward for exploring
    if new_cell_covered:
        reward += 1.0

        # Set milestone rewards for covering 25%, 50%, 75% and 100% of the map
        if total_covered_cells in [
            int(0.25 * coverable_cells),
            int(0.5 * coverable_cells),
            int(0.75 * coverable_cells),
            int(1.0 * coverable_cells),
        ]:
            reward += 1.0  # milestone bonus

    else:
        reward -= 0.1  # discourage idling or revisiting

    # Step penalty
    reward -= 0.01

    # Game over penalty
    if game_over and cells_remaining > 0:
        reward -= 5.0

    return reward
