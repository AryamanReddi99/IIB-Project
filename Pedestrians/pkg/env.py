import random
import numpy as np
from random import randint
from general import *


class PedEnv():
    """
    pedestrians environment
    "i" in agent_pos denotes which agent we are referring to
    """
    def __init__(self, GameConfig):
        self.done = 0
        self.reward_collision = -3
        self.reward_target = 3
        self.reward_move = -0.1
        self.done_list = []
        self.reward_list = []
        self.size = GameConfig.size
        self.num_agents = GameConfig.num_agents
        self.agent_size = GameConfig.agent_size
        self.agent_pos = GameConfig.pos_dict["agents"] # np arrays of agent positions
        self.target_pos = GameConfig.pos_dict["targets"] # np arrays of target positions

    def reset(self):
        self.done = 0
        self.done_list = []
        self.reward_list = []
        self.agent_pos = GameConfig.pos_dict["agents"] # np arrays of agent positions
        self.target_pos = GameConfig.pos_dict["targets"] # np arrays of target positions
        return *self.update_state(), self.reward_list, self.done_list, self.done

    def check_collision(self, pos1, pos2):
        """
        check if agent has collided with agent or target
        """
        if np.linalg.norm(pos1 - pos2) < self.agent_size:
            return True
        return False

    def return_reached(self):
        """
        indices of agents which reached target
        """
        list_reached = []
        for agent,posi in enumerate(self.agent_pos):
            if np.linalg.norm(posi - self.target_pos[agent]) < self.agent_size:
                list_reached.append(agent)
        return list_reached

    def return_collided(self):
        """
        return indices of agents which have collided
        """
        list_collided = []
        for agent1,posi in enumerate(self.agent_pos):
            for agent2,posj in enumerate(self.agent_pos):
                if agent1!=agent2:
                    if self.check_collision(posi,posj):
                        list_collided.append(agent1)
        return list_collided

    def update_state(self):
        # check boundaries
        for pos in self.agent_pos:
            pos[0] = bound(0,self.size-1,pos[0])
            pos[1] = bound(0,self.size-1,pos[1])

        return self.agent_pos, self.target_pos
    
    def __ex_reward(self):
        """
        reward function e^-x, x = proximity to target
        """
        return


    def action_space_sample(self):
        return

    def step(self,actions):
        self.done_list = []
        self.reward_list = []
        for agent,action in enumerate(actions):
            if action==0:
                pass
                # don't move
            elif action==1:
                # up
                self.agent_pos[agent][1] += speed
            elif action==2:
                # down
                self.agent_pos[agent][1] -= speed
            elif action==2:
                # left
                self.agent_pos[agent][0] -= speed
            elif action==2:
                # right
                self.agent_pos[agent][0] += speed

        collided = self.return_collided()
        reached = self.return_reached()

        for agent,pos in enumerate(self.agent_pos):
            if agent in collided:
                self.reward_list[agent] = self.reward_collision
            elif agent in reached:
                self.reward_list[agent] = self.reward_target
            else:
                self.reward_list[agent] = self.reward_move

        if len(collided) == self.num_agents:
            done = 1

        return *self.update_state(), self.reward_list, self.done_list, self.done
            

class PosConfigs():
    """
    contains ready-made agent/target position configs to be used
    """
    def __init__(self,size):
        self.size = size
        self.configs = [self.config_0(),self.config_1(),self.config_2]
    def config_0(self):
        """
        random placements on 8x8 possible locations, 2 agents
        """
        x = self.size/8
        y = self.size/8
        
        unique = False
        while not unique:
            # generate solutions until unieu ones obtained
            unique = True
            agent_1  = np.array([x*randint(1,8),y*randint(1,8)])
            agent_2  = np.array([x*randint(1,8),y*randint(1,8)])
            target_1 = np.array([x*randint(1,8),y*randint(1,8)])
            target_2 = np.array([x*randint(1,8),y*randint(1,8)])
            unique_list = [agent_1,agent_2,target_1,target_2] # need arrays to be unique so positions don't clash
            for i,arr in enumerate(unique_list):
                if len(list(filter(lambda x: x != arr, unique_list))) < len(unique_list) - 1:
                    unique = False
        return {
            "agents":agent_1,agent_2,
            "targets":target_1,target_2}

    def config_1(self):
        """
        crossing parallel pathways, 2 agents
        """
        x = self.size/8
        y = self.size/2
        agent_1  = np.array([x*3,y])
        agent_2  = np.array([x*5,y])
        target_1 = np.array([x*6,y])
        target_2 = np.array([x*2,y])
        return {
            "agents":agent_1,agent_2,
            "targets":target_1,target_2}

    def config_2(self):
        """
        crossing perpendicular pathways, 2 agents
        """
        x = self.size/4
        y = self.size/4
        agent_1  = np.array([x*2,y*3])
        agent_2  = np.array([x,y*2])
        target_1 = np.array([x*2,y])
        target_2 = np.array([x*3,y*2])
        return {
            "agents":   agent_1,agent_2,
            "targets":  target_1,target_2}
