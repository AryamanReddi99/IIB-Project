import collections
import numpy as np
from .general import *

################################# Main classes ##############################

class Q():
    """
    Class that executes all construction and training of a Q-Learner
    """
    def __init__(self, gameconfig, q_config):
        ## gameconfig
        self.i = gameconfig.i
        self.n = gameconfig.n
        self.games = gameconfig.games

        ## q_config
        self.mode = q_config.mode
        self.alpha = q_config.alpha
        self.gamma = q_config.gamma
        self.frac_random = q_config.frac_random
        self.final_epsilon = q_config.final_epsilon
        self.min_epsilon = q_config.min_epsilon
        self.reward_mode = q_config.reward_mode

        ## training
        self.trainable = True

        ## Buffers
        # stores last 2 sets of model-readable states
        self.state_buffer = collections.deque(maxlen=2)
        # stores experiences from one game
        self.game_buffer = []
        # stores experiences
        self.replay_buffer = collections.deque(maxlen=q_config.mem_max_size)

    def create_q(self):
        self.table = np.zeros([self.n+1,self.i])

    def update_state_buffer(self, i, t, turn):
        state = {
            "i":i,
            "t":t,
            "turn":turn
        }
        self.state_buffer.append(state)

    def update_game_buffer(self, action, reward, done):
        """
        Add new experience to game buffer
        Call update_state_buffer first
        """
        experience = Experience(self.state_buffer[-2], action, reward, self.state_buffer[-1], done)
        self.game_buffer.append(experience) 

    def act(self, i, t, game):
        # explore
        if random.uniform(0,1) < self._get_epsilon(game):
            action = self._action_space_sample(i,t)
        # exploit
        else:
            state = self.state_buffer[-1]
            qvals = self.table[state["t"]]
            action = np.argmax(qvals) + 1
        return action

    def train(self, move_total, reward_list):
        # Testing mode
        if self.mode == "testing":
            return
            
        # Update game buffer
        self._update_game_buffer(reward_list)

        # Update Q-values
        for exp in self.game_buffer:
            q_old = self.table[exp.state["t"]][exp.action-1]
            if not exp.done:
                q_next_max = np.max(self.table[exp.new_state["t"]])
                q_new = q_old + self.alpha * (exp.reward + self.gamma * q_next_max - q_old)
            else:
                q_new = q_old + self.alpha * (exp.reward - q_old)
            self.table[exp.state["t"]][exp.action-1] = q_new
        
        # Reset game buffer
        self.game_buffer = []

    def _update_game_buffer(self, reward_list):
        # Final
        if self.reward_mode==0:
            return self.game_buffer
        # Sparse
        if self.reward_mode==1:
            # single learner
            self.game_buffer[-1]._replace(reward=reward_list[self.game_buffer[-1].state["turn"]])
            # double learner
            if len(self.game_buffer>1) and (self.game_buffer[-1].state["turn"] != self.game_buffer[-2].state["turn"]):
                self.game_buffer[-2]._replace(reward=reward_list[self.game_buffer[-2 ].state["turn"]])
            return self.game_buffer
        # Full
        if self.reward_mode==2:
            self.game_buffer = list(map(lambda x: x._replace(reward=reward_list[x.state["turn"]]),self.game_buffer))
            return self.game_buffer

    def _get_epsilon(self, game):
        """
        Returns epsilon (random move chance) as decaying e^-x with first
        frac_random*games totally random

        game = current game number
        games = number of training games
        frac_random = fraction of games with epsilon = 1
        final_epsilon = epsilon after games have been played
        min_epsilon = minimum val of epsilon
        """
        if self.mode == "testing":
            self.epsilon = 0
        else:
            self.epsilon = bound(self.min_epsilon, 1, self.final_epsilon**((game - self.frac_random * \
            self.games) / (self.games * (1 - self.frac_random))))
        return self.epsilon

    def _action_space_sample(self,i,t):
        """
        Choose a random move
        """
        return random_play(i,t)

class QConfig():
    """
    mode:
    "training" -> works as normal
    "testing" -> doesn't explore (epsilon = 0)
    """
    def __init__(self,
                mode = "training",
                alpha = 0.4,
                gamma = 0.6,
                frac_random = 0.1,
                final_epsilon = 0.01,
                min_epsilon = 0.01,
                mem_max_size = 1000,
                reward_mode = 0
                ):
            # Training 
            self.mode = mode
            self.alpha = alpha
            self.gamma = gamma

            # Epsilon
            self.frac_random = frac_random
            self.final_epsilon = final_epsilon
            self.min_epsilon = min_epsilon
            self.mem_max_size = mem_max_size

            # Rewards
            # reward_mode:
            # 0: Final - only the rewards given by env
            # 1: Sparse - terminal move of either agent
            # 2: Full - terminal rewards propagate through whole game
            self.reward_mode = reward_mode

################################# External Functions/Classes ##############################

####################################### main() ####################################
def main():
    print("Finished")

if __name__ == "__main__":
    main()