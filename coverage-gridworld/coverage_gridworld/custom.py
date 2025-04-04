import numpy as np
import gymnasium as gym

"""
Feel free to modify the functions below and experiment with different environment configurations.
"""
def observation_space(env: gym.Env) -> gym.spaces.Space:
    """
    #---------------------------------OBSERVATION 1---------------------------------
    #Defines a 5x5x3 RGB observation space

    return gym.spaces.Box(low=0, high=255, shape=(5, 5, 3), dtype=np.uint8)
    """
    # ---------------------------------OBSERVATION 2---------------------------------
    # Defines a 7x7 cell-type observation space
    return gym.spaces.Box(low=0, high=1, shape=(7, 7,), dtype=np.uint8)


def observation(grid: np.ndarray):
    """
    #---------------------------------OBSERVATION 1---------------------------------
    #Returns a 5x5 grid around the agent

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
    """

    # ---------------------------------OBSERVATION 2---------------------------------

    # Always return something, even if broken
    if grid is None:
        return np.zeros((7,7,), dtype=np.uint8)

    try:
        # get position of agent
        y, x = np.argwhere(np.all(grid == [160, 161, 161], axis=-1))[0]
        padded = np.pad(
            grid, ((3, 3), (3, 3), (0, 0)), mode="constant", constant_values=5
        )
        # have a 7x7 window of visibility surrounding the agent
        window = padded[y : y + 7, x : x + 7]
        translated = [[],[],[],[],[],[],[]] # the info will be translated to 7x7 rather than 7x7x3
        flattened = list(np.ravel(window))  # get all vals in order
        while len(flattened) > 0:
            for i in range(7):
                while len(translated[i]) < 7:
                    r = flattened.pop(0)
                    g = flattened.pop(0)
                    b = flattened.pop(0)
                    
                    #if black
                    if (r == 0 and g == 0 and b == 0):
                        translated[i].append(0) # need to cover
                    #if red
                    elif (r == 255 and g == 0 and b == 0):
                        translated[i].append(1) # need to cover but danger area
                    #if white, grey
                    elif ((r == 255 and g == 255 and b == 255) or 
                          (r == 160 and g == 161 and b == 161)):
                        translated[i].append(2) # already covered but walkable
                    # light-red
                    elif (r == 255 and g == 127 and b == 127):
                        translated[i].append(3) # already covered but walkable dangerously
                    #if brown, green, or padding
                    else:
                        translated[i].append(4)  # can't go there

        return np.array(translated)
    except:
        return np.zeros((7,7,), dtype=np.uint8)



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

    """
    #---------------------------------REWARD 1---------------------------------
    reward = 0.0

    # Base reward for exploring
    if new_cell_covered:
        reward += 1.0

        # Set milestone rewards for covering 25%, 50%, 75% and 100% of the map
        if total_covered_cells == int(0.25 * coverable_cells):
            reward += 1.0
        elif total_covered_cells == int(0.5 * coverable_cells):
            reward += 2.0
        elif total_covered_cells == int(0.75 * coverable_cells):
            reward += 3.0

        # Proportional reward every step
        coverage_ratio = total_covered_cells / coverable_cells
        reward += 0.5 * coverage_ratio

    else:
        reward -= 0.1  # discourage idling or revisiting



    if cells_remaining == 0:
        reward += 10.0 + steps_remaining * 0.1  # bonus for finishing fast
    if game_over:
        reward -= 10.0  # punished for failing before completion

    return reward"""

    # ---------------------------------REWARD 2---------------------------------
    # Penalize the agent if it is in the future FOV of the enemies
    if len(str(agent_pos)) == 1:
        unflattened_agent_pos = (0, agent_pos)
    else:
        unflattened_agent_pos = (int(str(agent_pos)[0]), int(str(agent_pos)[1]))

    # for each enemy, calculate all the fov's it can have (excluding its current fov)
    future_fovs = []

    for enemy in enemies:
        for dir in range(0, 4):
            x = enemy.x
            y = enemy.y
            orientation = enemy.orientation

            for i in range(1, 5):
                if orientation == i:  # condition to exclude current fov
                    break
                if dir == 0:  # LEFT
                    fov_row, fov_col = y, x - i
                elif dir == 1:  # DOWN
                    fov_row, fov_col = y + i, x
                elif dir == 2:  # RIGHT
                    fov_row, fov_col = y, x + i
                else:  # UP
                    fov_row, fov_col = y - i, x

                future_fovs.append((fov_col, fov_row))

        # Reward for game_over with no cells remaining and penalize if agent is killed by ghost
        if cells_remaining == 0:
            return 50

        if game_over:
            return -50.0

        if steps_remaining == 0:
            return -50.0

        # calculate the total accumulated reward
        reward = 0

        if new_cell_covered:
            reward += 3.0

            # further encourage the agent to discover cells near the enemies
            if unflattened_agent_pos in future_fovs:
                reward += 4.0

        else:
            reward -= 1.0

        return reward

    """
    #---------------------------------REWARD 3---------------------------------
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

    return reward"""
