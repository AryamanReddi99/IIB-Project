from collections import OrderedDict
from random import randint

import numpy as np


def bound(low, high, value):
    """
    Bounds value between low, high
    """
    return max(low, min(high, value))


def float2mat(pos, size):
    """
    Converts position in np.array x-y format into square matrix
    e.g. [0,0] in 2x2 grid -> [0 0]
                              [1 0]
    """
    if len(list(pos)) > 2:
        # wall
        mat = np.zeros([size, size])
        for wall_element in pos:
            mat += float2mat(wall_element, size)
    else:
        # point
        try:
            x = int(pos[0])
            y = int(pos[1])
            mat = np.zeros([size, size])
        except:
            raise TypeError("position must be integer")

        # if outside bounds, all-0 matrix
        if x >= 0 and x <= size - 1 and y >= 0 and y <= size - 1:
            mat[size - 1 - y, x] = 1
    return mat

def float2mat_anti_target(pos, size):
    """
    Converts a wall float position array into an anti-target (the opposing walls)
    """
    anti_target_walls = ["top", "bottom", "left", "right"]
    target_wall = which_wall(pos)
    anti_target_walls.remove(target_wall)
    mat = np.zeros([size, size])
    for wall in anti_target_walls:
        mat += float2mat(create_wall(wall, size), size)
    return np.clip(mat, a_min=0, a_max=1)

def mat2float(arr):
    """
    converts point/wall position in matrix form back to float form
    """
    env_size, _ = arr.shape
    indices = np.argwhere(arr)  # indices of non-zero points
    flipped_indices = np.flip(indices, axis=1)  # flip x and y coords
    if len(flipped_indices) > 1:
        # wall or walls
        if len(flipped_indices) > env_size:
            # walls
            # just return nothing, don't need it yet
            return
        else:
            # wall
            flipped_indices[:, 1] = (
            env_size - 1 - flipped_indices[:, 1]
        )  # flip y about centre
        return flipped_indices
    elif len(flipped_indices) == 1:
        # point
        [[x, ny]] = flipped_indices  # y is flipped with np array
        return np.array([x, env_size - 1 - ny]).astype(int)
    else:
        # not on grid
        return np.array([-1, -1]).astype(int)

def float2pygame(pos, size):
    """
    Converts position in np.array x-y format into pygame format (flipped on y-axis)
    e.g. [50,99] in 100x100 grid -> [50,0]
    """
    if len(list(pos)) > 2:
        # wall
        return np.flip(pos, axis=0)
    else:
        # point
        try:
            x = int(pos[0])
            y = int(pos[1])
        except:
            raise TypeError("position must be integer")
        return np.array([x, size - 1 - y])


def arr2stack(arr):
    """
    converts from ndarray list of channels (channels,size,size) to
    model-friendly RGB-stack format (1,size,size,channels)
    """
    channels, env_size, _ = arr.shape
    return np.transpose(arr, (1, 2, 0)).reshape(1, env_size, env_size, channels)


def stack2arr(arr):
    """
    converts from model-friendly RGB stack format to ndarray list of
    separate channels
    """
    return np.transpose(np.squeeze(arr), (2, 0, 1))


def which_wall(pos):
    """
    Given pos array, determine which wall we're looking at
    """
    # [x,y] -> [bool, bool] for fixed wall axis
    # e.g. vertical wall    -> [True, False]
    # e.g. horizontal wall  -> [False, True]

    env_size, _ = pos.shape
    wall_alignment = np.equal(pos, np.flip(pos, axis=0)).all(axis=0)
    # Vertical walls
    if wall_alignment[0]:
        # Left
        if pos[0][0] == 0:
            return "left"
        # Right
        elif pos[0][0] == env_size - 1:
            return "right"
    # Horizontal walls
    elif wall_alignment[1]:
        # Bottom
        if pos[0][1] == 0:
            return "bottom"
        # Top
        elif pos[0][1] == env_size - 1:
            return "top"
    raise ValueError("Invalid wall!")


def create_wall(wall, size):
    """
    returns a wall target for a given env size
    wall format is list of 2-D numpy arrays with coordinates of wall
    wall coordinates must be at edges of environment
    """
    if wall == "top":
        wall = [np.array([i, size - 1]) for i in range(0, size)]
    elif wall == "bottom":
        wall = [np.array([i, 0]) for i in range(0, size)]
    elif wall == "left":
        wall = [np.array([0, i]) for i in range(0, size)]
    elif wall == "right":
        wall = [np.array([size - 1, i]) for i in range(0, size)]
    else:
        raise ValueError("Incorrect argument for wall")
    return np.array(wall)


def pretty_state(state):
    """
    coverts a model-friendly state into something readable
    """
    state_list_format = stack2arr(state)
    pretty_list = list(map(lambda x: mat2float(x), state_list_format))
    pretty_list[1] = which_wall(pretty_list[1])  # convert wall
    return pretty_list


def pretty_experience(experience):
    """
    make model-readable experience human-readable
    """
    state, action, reward, new_state, done = experience
    pretty_experience = OrderedDict()
    pretty_experience["state"] = pretty_state(state)
    pretty_experience["action"] = action
    pretty_experience["reward"] = reward
    pretty_experience["new_state"] = pretty_state(new_state)
    pretty_experience["done"] = done
    return pretty_experience


class _PosConfig:
    """
    Internal class in GameConfig()
    Contains ready-made agent/target position configs to be used

    2-player:
    0-9: ball targets
    0 - random non-conflicting placements in 8x8 grid
    1 - crossing parallel pathways
    2 - crossing perpendicular pathways

    10-19: wall targets
    10 - random (tba)
    11 - crossing parallel pathways (horizontal)
    12 - crossing perpendicular pathways
    13 - crossing parallel pathways (vertical)

    3-player:
    100 - left, right, top
    """

    def __init__(self, size):
        self.size = size
        self.configs = {
            0: self.config_0(),
            1: self.config_1(),
            2: self.config_2(),
            11: self.config_11(),
            12: self.config_12(),
            13: self.config_13(),
            14: self.config_14(),
            15: self.config_15(),
            100: self.config_100(),
        }

    def config_0(self):
        """
        random placements on 8x8 possible locations, 2 agents
        """
        x = self.size / 8
        y = self.size / 8

        unique = False
        while not unique:
            # generate solutions until unique ones obtained
            unique = True
            agent_1 = np.array([x * randint(1, 8), y * randint(1, 8)])
            agent_2 = np.array([x * randint(1, 8), y * randint(1, 8)])
            target_1 = np.array([x * randint(1, 8), y * randint(1, 8)])
            target_2 = np.array([x * randint(1, 8), y * randint(1, 8)])
            # need arrays to be unique so positions don't clash
            unique_list = [agent_1, agent_2, target_1, target_2]
            for i, arr in enumerate(unique_list):
                if len(list(filter(lambda x: (x == arr).all(), unique_list))) > 1:
                    unique = False
        return {"agents": [agent_1, agent_2], "targets": [target_1, target_2]}

    def config_1(self):
        """
        crossing parallel pathways, 2 agents
        """
        x = self.size / 8
        y = self.size / 2
        agent_1 = np.array([x * 3, y])
        agent_2 = np.array([x * 5, y])
        target_1 = np.array([x * 6, y])
        target_2 = np.array([x * 2, y])
        return {"agents": [agent_1, agent_2], "targets": [target_1, target_2]}

    def config_2(self):
        """
        crossing perpendicular pathways, 2 agents
        """
        x = self.size / 4
        y = self.size / 4
        agent_1 = np.array([x * 2, y * 3])
        agent_2 = np.array([x, y * 2])
        target_1 = np.array([x * 2, y])
        target_2 = np.array([x * 3, y * 2])
        return {"agents": [agent_1, agent_2], "targets": [target_1, target_2]}

    def config_11(self):
        """
        crossing horizontal parallel pathways, 2 agents
        """
        y = self.size / 2
        agent_1 = np.array([0, y])
        agent_2 = np.array([self.size - 1, y])
        target_1 = create_wall("right", self.size)
        target_2 = create_wall("left", self.size)
        return {"agents": [agent_1, agent_2], "targets": [target_1, target_2]}

    def config_12(self):
        """
        crossing perpendicular paths, 2 agents
        left-top
        """
        x = self.size / 2
        y = self.size / 2
        agent_1 = np.array([0, y - 1])
        agent_2 = np.array([x, self.size - 1])
        target_1 = create_wall("right", self.size)
        target_2 = create_wall("bottom", self.size)
        return {"agents": [agent_1, agent_2], "targets": [target_1, target_2]}

    def config_13(self):
        """
        crossing vertical parallel pathways, 2 agents
        """
        x = self.size / 2
        agent_1 = np.array([x, self.size - 1])
        agent_2 = np.array([x, 0])
        target_1 = create_wall("bottom", self.size)
        target_2 = create_wall("top", self.size)
        return {"agents": [agent_1, agent_2], "targets": [target_1, target_2]}

    def config_14(self):
        """
        crossing perpendicular paths, 2 agents
        bottom-right
        """
        x = self.size / 2
        y = self.size / 2
        agent_1 = np.array([x, 0])
        agent_2 = np.array([self.size - 1, y])
        target_1 = create_wall("top", self.size)
        target_2 = create_wall("left", self.size)
        return {"agents": [agent_1, agent_2], "targets": [target_1, target_2]}

    def config_15(self):
        """
        crossing perpendicular paths, 2 agents
        left-bottom
        """
        x = self.size / 2
        y = self.size / 2
        agent_1 = np.array([0, y])
        agent_2 = np.array([x, 0])
        target_1 = create_wall("right", self.size)
        target_2 = create_wall("top", self.size)
        return {"agents": [agent_1, agent_2], "targets": [target_1, target_2]}

    def config_100(self):
        """
        left, right, top, 3 agents
        """
        x = self.size / 2
        y = self.size / 2
        agent_1 = np.array([1, y])
        agent_2 = np.array([self.size - 2, y])
        agent_3 = np.array([x, self.size - 2])
        target_1 = create_wall("right", self.size)
        target_2 = create_wall("left", self.size)
        target_3 = create_wall("bottom", self.size)
        return {
            "agents": [agent_1, agent_2, agent_3],
            "targets": [target_1, target_2, target_3],
        }


class GameConfig:
    def __init__(
        self,
        env_size=256,
        config=14,
        speed=4,
        num_agents=2,
        agent_size=8,
        channels=4,
        num_actions=5,
        games=100,
        max_game_length=50,
        doom=False,
    ):

        # Starting Configurations
        self.posconfig = _PosConfig(env_size)
        self.pos_dict_initial = self.posconfig.configs[config].copy()

        # Game Parameters
        self.env_size = env_size
        self.config = config
        self.speed = speed
        self.num_agents = num_agents
        self.agent_size = agent_size
        self.channels = channels
        self.num_actions = num_actions
        self.games = games
        self.max_game_length = max_game_length

        # Assertions
        assert self.speed <= 2 * np.sqrt(2) * self.agent_size

        # Doom
        self.doom = doom


####################################### main() ####################################


def main():
    posconfig = _PosConfig(64)
    gameconfig = GameConfig(env_size=256, config=1, speed=4, num_agents=2, agent_size=5)
    print(posconfig.configs)


if __name__ == "__main__":
    main()
