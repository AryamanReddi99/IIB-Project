import os
import time
import random
import pygame
import pygame.freetype
from win32api import GetSystemMetrics
from pygame.locals import *
from pkg.general import *
from pkg.env import *

class Window():
    def __init__(self, screenconfig, gameconfig):
        """
        Window class for pygame drawing. Draw env, border, text on env, border,
        text screen. Then blit to fakescreen unscaled. Then blit and scale fake_screen
        to screen
        """

        # Initialise
        self.headless = screenconfig.headless
        if self.headless:
            return
        print("Initialising Display...")
        pygame.init()
        os.chdir(os.path.dirname(__file__))

        # Controls
        self.pause = False
        self.next_frame = False

        # Relative Sizes
        self.border_size = screenconfig.border_size
        self.env_size = gameconfig.env_size
        self.num_agents = gameconfig.num_agents
        self.agent_size = gameconfig.agent_size
        self.config = gameconfig.config
        self.window_size = (
            2 * (self.env_size + 2 * self.border_size), 
            self.env_size + 2 * self.border_size)

        # Visuals
        self.invisible_agents = [False for _ in range(self.num_agents)]
        self.blue = (0, 0, 255)
        self.red = (255, 0, 0)
        self.green = (0, 200, 0)
        self.yellow = (255, 215, 0)
        self.purple = (102, 0, 204)
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.colours = [
            self.blue,
            self.red,
            self.green,
            self.yellow,
            self.purple,
            self.white]

        # Text
        sep = os.path.sep
        if gameconfig.env_size == 256:
            font_size = 20
        elif gameconfig.env_size == 128:
            font_size = 10
        elif gameconfig.env_size == 64:
            font_size = 5
        else:
            font_size = 25
        font_1 = pygame.freetype.SysFont("constantia", font_size)
        font_2 = pygame.freetype.SysFont("couriernew", 10)
        font_3 = pygame.freetype.Font(f"Doom{sep}DooM.ttf", 22)
        self.font = font_1

        # Create screens and surfaces
        self.screen = pygame.display.set_mode(
            self.window_size, DOUBLEBUF | RESIZABLE)
        self.fake_screen = self.screen.copy()

        self.border_screen = pygame.surface.Surface(
            (self.env_size + 2 * self.border_size,
             self.env_size + 2 * self.border_size))
        self.env_screen = pygame.surface.Surface(
            (self.env_size, self.env_size))
        self.text_screen = pygame.surface.Surface(
            (self.env_size + 2 * self.border_size,
             self.env_size + 2 * self.border_size))

        # Doom
        self.doom = gameconfig.doom
        if self.doom:
            self.cacodemon_left = pygame.image.load(f'Doom{sep}caco_left.png')
            self.cacodemon_right = pygame.image.load(f'Doom{sep}caco_right.png')
            self.font = font_3
            self.agent_size = 16

    def _draw_border(self):
        pygame.draw.rect(
            self.border_screen,
            self.purple,
            pygame.Rect(
                0,
                0,
                self.env_size + self.border_size * 2,
                self.env_size + self.border_size * 2))
        pygame.draw.rect(
            self.border_screen,
            self.black,
            pygame.Rect(
                self.border_size,
                self.border_size,
                self.env_size,
                self.env_size))

    def _draw_agents(self, display_info):
        agent_pos = display_info.agent_pos
        if not self.doom:
            for agent, pos in enumerate(agent_pos):
                pygame.draw.circle(
                    self.env_screen,
                    self.colours[agent],
                    pos,
                    self.agent_size)
        else:
            (caco_width, caco_height) = self.cacodemon_left.get_rect().size
            for agent, pos in enumerate(agent_pos):
                if agent==0:
                    self.env_screen.blit(self.cacodemon_right, (pos[0] - 0.5*caco_width, pos[1] - 0.5*caco_height))
                elif agent==1:
                    self.env_screen.blit(self.cacodemon_left, (pos[0] - 0.5*caco_width, pos[1] - 0.5*caco_height))

    def _draw_targets(self, target_pos):
        if self.config < 10:
            frac = 0.25  # fraction of radius used for outer circle & inner space
            for target, pos in enumerate(target_pos):
                pygame.draw.circle(self.env_screen, self.yellow,
                                pos, self.agent_size * (1 + frac))
                pygame.draw.circle(
                    self.env_screen,
                    self.colours[target],
                    pos,
                    self.agent_size)
                pygame.draw.rect(
                    self.env_screen,
                    self.yellow,
                    pygame.Rect(pos[0] -2 *self.agent_size,
                        pos[1] -frac *0.5 *self.agent_size,
                        4 *self.agent_size,
                        frac *self.agent_size))
                pygame.draw.rect(
                    self.env_screen,
                    self.yellow,
                    pygame.Rect(
                        pos[0] - frac * 0.5 * self.agent_size, 
                        pos[1] - 2 * self.agent_size, 
                        frac * self.agent_size, 
                        4 * self.agent_size))
        elif self.config < 20:
            return

    def _draw_text(self, display_info):
        """
        Draw text info 
        """
        # Extract data from display_info dict
        action_list = display_info.action_list
        reward_list = display_info.reward_list
        done_list = display_info.done_list
        collided_list = display_info.collided_list
        reached_list = display_info.reached_list
        breached_list = display_info.breached_list
        done = display_info.done
        game = display_info.game
        move = display_info.move

        # Vertical space between text statements
        text_spacing = 20

        # Write game and move info
        self.font.render_to(self.text_screen, (5, 5), "Game: " + str(game), self.white)
        self.font.render_to(self.text_screen, (5, 5 + text_spacing), "Move: " + str(move), self.white)

        # Write states of agents
        agent_states = [None for _ in range(len(collided_list))]
        for agent in range(len(collided_list)):
            if collided_list[agent]:
                agent_states[agent] = "has collided!"
            elif breached_list[agent]:
                agent_states[agent] = "got lost!"
            elif reached_list[agent]:
                agent_states[agent] = "has reached!"
            else:
                agent_states[agent] = "is searching..."

        # Write to Screen
        self.font.render_to(self.text_screen, (5, 5 + 2*text_spacing), "Agent 1 " + agent_states[0], self.white)
        self.font.render_to(self.text_screen, (5, 5 + 3*text_spacing), "Agent 2 " + agent_states[1], self.white)

        # Pause and Next Frame
        self.font.render_to(self.text_screen, (5, self.env_size  + 2*self.border_size - 2*text_spacing), "Press p to pause/unpause", self.white)
        self.font.render_to(self.text_screen, (5, self.env_size  + 2*self.border_size - text_spacing), "Press n to run next frame", self.white)
    
    def _draw_to_fake(self):
        self.fake_screen.blit(self.border_screen, (self.env_size + 2 * self.border_size, 0))
        self.fake_screen.blit(self.env_screen,(self.env_size + 3 *self.border_size,self.border_size))
        self.fake_screen.blit(self.text_screen, (0, 0))

    def _clear_screens(self):
        self.border_screen.fill('black')
        self.env_screen.fill('black')
        self.text_screen.fill('black')

    def _refresh_display(self):
        """
        Checks for screen quit, pause, or resize
        Draws to fake screen, blits fake screen to screen,
        clears all hideen screens, flips display
        """
        self.next_frame = False
        # Check Events
        for event in pygame.event.get():
            # QUIT event
            if event.type == pygame.QUIT:
                pygame.display.quit()
            # PAUSE event
            elif event.type == pygame.KEYUP:
                if event.key==K_p:
                    self.pause = True
            # RESIZE event
            elif event.type == VIDEORESIZE:
                screen_width = GetSystemMetrics(0)
                screen_height = GetSystemMetrics(1)
                if min(event.size) > 0.9 * screen_height:  # going fullscreen
                    self.screen = pygame.display.set_mode(
                        (screen_width, int(screen_width * 0.5)), DOUBLEBUF | RESIZABLE)
                else:
                    self.screen = pygame.display.set_mode(
                        (min(event.size) * 2, min(event.size)), DOUBLEBUF | RESIZABLE)
        # WHEN PAUSED
        while self.pause and not self.next_frame:
            for event in pygame.event.get():
                if event.type==KEYUP:
                    if event.key==K_n:
                        # Run one frame
                        self.next_frame = True
                    if event.key==K_p:
                        # Unpause
                        self.pause = False

        self._draw_to_fake()
        self.screen.blit(pygame.transform.scale(self.fake_screen, self.screen.get_rect().size), (0, 0))
        self._clear_screens()
        pygame.display.flip()  # flip the 2 buffers (visible one and the one being drawn)

    def display(self, display_info):
        if self.headless:
            return

        # Display_info is dict with relevant data
        agent_pos = display_info.agent_pos
        target_pos = display_info.target_pos


        # Update everything on screen
        self._draw_border()
        self._draw_targets(target_pos)
        self._draw_agents(display_info)
        self._draw_text(display_info)
        self._refresh_display()

class ScreenConfig():
    """
    Stores config information about pygame display
    """

    def __init__(self, headless=False, border_size=10):
        self.headless = headless
        self.border_size = border_size

class DisplayInfo():
    def __init__(self,
            agent_pos, 
            target_pos, 
            action_list,
            reward_list,
            done_list, 
            collided_list, 
            reached_list, 
            breached_list,
            done,
            game,
            move):
        
        self.agent_pos = agent_pos
        self.target_pos = target_pos
        self.action_list = action_list
        self.reward_list = reward_list 
        self.done_list = done_list
        self.collided_list = collided_list
        self.reached_list = reached_list
        self.breached_list = breached_list
        self.done = done
        self.game = game
        self.move = move

def main():
    gameconfig = GameConfig(
        env_size=256,
        config=1,
        speed=4,
        num_agents=2,
        agent_size=8,
        channels=4,
        num_actions=5)
    screenconfig = ScreenConfig(
        border_size=10)
    window = Window(screenconfig, gameconfig)

    i = 0
    while True:
        i += 1
        agent_pos = [[random.randint(0, 255), random.randint(0, 255)], [
            random.randint(0, 255), random.randint(0, 255)]]
        target_pos = [[random.randint(0, 255), random.randint(0, 255)], [
            random.randint(0, 255), random.randint(0, 255)]]

        display_info = {
            "agent_pos": agent_pos,
            "target_pos": target_pos,
            "text": i
        }
        window.display(display_info=display_info)

        time.sleep(0.3)


if __name__ == "__main__":
    main()
