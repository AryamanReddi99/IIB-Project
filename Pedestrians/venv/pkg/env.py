from random import randint

import numpy as np

from .general import *


class PedEnv:
    """
    pedestrians environment
    "i" in agent_pos denotes which agent we are referring to
    """

    def __init__(self, gameconfig):

        # np arrays of agent positions
        self.original_agent_pos = np.copy(gameconfig.pos_dict_initial["agents"])

        # np arrays of target positions
        self.original_target_pos = np.copy(gameconfig.pos_dict_initial["targets"])

        # game params
        self.env_size = gameconfig.env_size
        self.config = gameconfig.config
        self.speed = gameconfig.speed
        self.num_agents = gameconfig.num_agents
        self.agent_size = gameconfig.agent_size
        self.num_actions = gameconfig.num_actions
        self.max_episode_length = gameconfig.max_episode_length
        self.reward_target = gameconfig.reward_target
        self.reward_death = gameconfig.reward_death
        self.reward_move = gameconfig.reward_move

        # episode params
        self.done = 0

    def _check_collision_agents(self, pos1, pos2):
        """
        check if agents have collided
        """
        pos1_mat = float2mat_agent(pos1, self.env_size, self.agent_size)
        pos2_mat = float2mat_agent(pos2, self.env_size, self.agent_size)
        # superpose agent images. If they overlap, they're in contact
        if np.max(pos1_mat + pos2_mat) == 2:
            return True
        return False

    def _check_collision_wall(self, pos1, pos2):
        """
        check if agent has collided with wall
        """
        if np.linalg.norm(pos1 - pos2) < 2 * self.agent_size:
            return True
        return False

    def _return_reached(self):
        """
        indices of agents which reached target
        """
        if self.config < 10:
            # ball targets
            for agent, pos in enumerate(self.agent_pos):
                self.reached_list[agent] = self._check_collision_ball(
                    pos, self.target_pos[agent]
                )
        elif self.config < 20:
            # Check if correct wall has been passed
            for agent, pos in enumerate(self.agent_pos):
                target = self.target_pos[agent]
                wall = which_wall(target)
                if wall == "left":
                    self.reached_list[agent] = pos[0] < 0
                elif wall == "right":
                    self.reached_list[agent] = pos[0] > self.env_size - 1
                elif wall == "bottom":
                    self.reached_list[agent] = pos[1] < 0
                elif wall == "top":
                    self.reached_list[agent] = pos[1] > self.env_size - 1
        return self.reached_list

    def _return_collided(self):
        """
        Return Indices of Agents Which Have Collided
        """
        for agent1, posi in enumerate(self.agent_pos):
            for agent2, posj in enumerate(self.agent_pos):
                if agent1 != agent2:
                    if self._check_collision_agents(posi, posj):
                        self.collided_list[agent1] = True
        return self.collided_list

    def _return_breached(self):
        """
        Return indices of Agents beyond boundaries not in reached_list (dead)
        """
        for agent, pos in enumerate(self.agent_pos):
            # Check for breach
            if (pos < 0).any() or (pos > self.env_size - 1).any():
                # Check if reached
                if not self.reached_list[agent]:
                    self.breached_list[agent] = True
        return self.breached_list

    def _update_state(self):
        """
        Check Boundaries
        """
        if self.config < 10:
            # Ball targets
            for pos in self.agent_pos:
                pos[0] = bound(0, self.env_size - 1, pos[0])
                pos[1] = bound(0, self.env_size - 1, pos[1])
        elif self.config < 20:
            # Wall targets
            for agent, pos in enumerate(self.agent_pos):
                if self.breached_list[agent]:
                    pos[0] = self.original_agent_pos[agent][0]
                    pos[1] = self.original_agent_pos[agent][1] + self.env_size * 2
        return self.agent_pos, self.target_pos

    def reset(self):
        # Done Agents
        self.done = 0
        self.done_list = [False for _ in range(self.num_agents)]
        self.collided_list = [False for _ in range(self.num_agents)]
        self.reached_list = [False for _ in range(self.num_agents)]
        self.breached_list = [False for _ in range(self.num_agents)]

        # Rewards
        self.reward_list = [None for _ in range(self.num_agents)]

        # Agent/Target Positions
        self.agent_pos = np.copy(self.original_agent_pos)
        self.target_pos = np.copy(self.original_target_pos)

        return (
            *self._update_state(),
            self.reward_list,
            self.done_list,
            self.collided_list,
            self.reached_list,
            self.breached_list,
            self.done,
        )

    def action_space_sample(self):
        """
        Return Random Action
        """
        return randint(0, self.num_actions - 1)

    def step(self, actions):
        for agent, action in enumerate(actions):
            if action == 0:
                # don't move
                pass
            elif action == 1:
                # up
                self.agent_pos[agent][1] += self.speed
            elif action == 2:
                # down
                self.agent_pos[agent][1] -= self.speed
            elif action == 3:
                # left
                self.agent_pos[agent][0] -= self.speed
            elif action == 4:
                # right
                self.agent_pos[agent][0] += self.speed

        self._update_state()
        self._return_collided()
        self._return_reached()
        self._return_breached()

        for agent, pos in enumerate(self.agent_pos):
            if self.collided_list[agent]:
                self.reward_list[agent] = self.reward_death
                self.done_list[agent] = True
            elif self.breached_list[agent]:
                self.reward_list[agent] = self.reward_death
                self.done_list[agent] = True
            elif self.reached_list[agent]:
                self.reward_list[agent] = self.reward_target
                self.done_list[agent] = True
            elif actions[agent] == 0:
                self.reward_list[agent] = self.reward_move
            else:
                self.reward_list[agent] = self.reward_move

        if all(self.done_list):
            self.done = 1

        return (
            *self._update_state(),
            self.reward_list,
            self.done_list,
            self.collided_list,
            self.reached_list,
            self.breached_list,
            self.done,
        )


####################################### main() ####################################


def main():
    gameconfig = GameConfig(env_size=256, config=1, speed=4, num_agents=2, agent_size=5)
    env = PedEnv(gameconfig)
    print("Finished")


if __name__ == "__main__":
    main()
