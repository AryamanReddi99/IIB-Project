import os
import time
import random
import pygame
import pygame.freetype

# from win32api import GetSystemMetrics
from pygame.locals import *
from .general import *


class Window:
    """
    Window class for pygame drawing. Draw env, border, text on env, text screen.
    Then blit to fakescreen unscaled. Then blit and scale fake_screen to screen
    """

    def __init__(self, screenconfig, gameconfig):
        ## screenconfig
        self.headless = screenconfig.headless
        self.border_size = screenconfig.border_size
        self.font_size = screenconfig.font_size
        self.font = screenconfig.font
        self.window_width = screenconfig.window_width
        self.text_spacing = screenconfig.text_spacing

        ## gameconfig
        self.env_size = gameconfig.env_size
        self.num_agents = gameconfig.num_agents
        self.agent_size = gameconfig.agent_size
        self.doom = gameconfig.doom

        # Initialise
        if self.headless:
            # Don't create Pygame display
            return
        print("Initialising Display...")
        pygame.init()

        # Path
        sep = os.path.sep
        pkg_path = f"..{sep}..{sep}..{sep}pkg{sep}"

        # Controls
        self.pause = False
        self.next_frame = False

        ## Relative Sizes
        # true_screen has the true env size + agent sizes
        # scaled_screen has scaled env + agents, and is the actual size of the display we want
        # display_screen has same size as fake_screen
        self.true_size = (
            2 * (self.env_size + 2 * self.border_size),
            self.env_size + 2 * self.border_size,
        )
        self.display_size = (self.window_width, int(0.5 * self.window_width))

        # Visuals
        self.invisible_agents = [False for _ in range(self.num_agents)]
        self.blue = (0, 0, 255)
        self.red = (255, 0, 0)
        self.green = (0, 200, 0)
        self.yellow = (255, 215, 0)
        self.purple = (102, 0, 204)
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.colours = [self.blue, self.red, self.green]

        # Text
        self.font = pygame.freetype.SysFont(self.font, self.font_size)

        ## Create screens and surfaces
        # screens
        self.display_screen = pygame.display.set_mode(
            size=self.display_size, flags=DOUBLEBUF | RESIZABLE
        )
        self.scaled_screen = self.display_screen.copy()
        self.true_screen = pygame.surface.Surface(size=self.true_size)

        # surfaces
        self.border_screen = pygame.surface.Surface(
            (self.env_size + 2 * self.border_size, self.env_size + 2 * self.border_size)
        )
        self.env_screen = pygame.surface.Surface((self.env_size, self.env_size))
        self.text_screen = pygame.surface.Surface(
            (0.5 * self.window_width, 0.5 * self.window_width)
        )

        # Doom
        self.doom = gameconfig.doom
        if self.doom:
            self.cacodemon_left = pygame.image.load(
                pkg_path + f"Doom{sep}caco_left.png"
            )
            self.cacodemon_right = pygame.image.load(
                pkg_path + f"Doom{sep}caco_right.png"
            )
            font_doom = pygame.freetype.Font(pkg_path + f"Doom{sep}DooM.ttf", 15)
            self.font = font_doom
            self.agent_size = 16

    def _draw_border(self):
        """Draw border on border_screen"""
        pygame.draw.rect(
            self.border_screen,
            self.purple,
            pygame.Rect(
                0,
                0,
                self.env_size + self.border_size * 2,
                self.env_size + self.border_size * 2,
            ),
        )
        pygame.draw.rect(
            self.border_screen,
            self.black,
            pygame.Rect(
                self.border_size, self.border_size, self.env_size, self.env_size
            ),
        )

    def _draw_agents(self, display_info):
        """Draw agents on env_screen"""
        agent_pos = display_info.agent_pos
        if not self.doom:
            for agent, pos in enumerate(agent_pos):
                pygame.draw.circle(
                    self.env_screen, self.colours[agent], pos, self.agent_size
                )
        else:
            (caco_width, caco_height) = self.cacodemon_left.get_rect().size
            for agent, pos in enumerate(agent_pos):
                if agent == 0:
                    self.env_screen.blit(
                        self.cacodemon_right,
                        (pos[0] - 0.5 * caco_width, pos[1] - 0.5 * caco_height),
                    )
                elif agent == 1:
                    self.env_screen.blit(
                        self.cacodemon_left,
                        (pos[0] - 0.5 * caco_width, pos[1] - 0.5 * caco_height),
                    )

    def _draw_targets(self, target_pos):
        if self.config < 10:
            frac = 0.25  # fraction of radius used for outer circle & inner space
            for target, pos in enumerate(target_pos):
                pygame.draw.circle(
                    self.env_screen, self.yellow, pos, self.agent_size * (1 + frac)
                )
                pygame.draw.circle(
                    self.env_screen, self.colours[target], pos, self.agent_size
                )
                pygame.draw.rect(
                    self.env_screen,
                    self.yellow,
                    pygame.Rect(
                        pos[0] - 2 * self.agent_size,
                        pos[1] - frac * 0.5 * self.agent_size,
                        4 * self.agent_size,
                        frac * self.agent_size,
                    ),
                )
                pygame.draw.rect(
                    self.env_screen,
                    self.yellow,
                    pygame.Rect(
                        pos[0] - frac * 0.5 * self.agent_size,
                        pos[1] - 2 * self.agent_size,
                        frac * self.agent_size,
                        4 * self.agent_size,
                    ),
                )
        elif self.config < 20:
            return

    def _draw_text(self, display_info):
        """
        Draw text info
        """
        ## Extract data from display_info object
        # action_list = display_info.action_list
        # reward_list = display_info.reward_list
        # done_list = display_info.done_list
        collided_list = display_info.collided_list
        reached_list = display_info.reached_list
        breached_list = display_info.breached_list
        # done = display_info.done
        game = display_info.game
        move = display_info.move

        num_agents = len(collided_list)

        # Agent states
        agent_states = [None for _ in range(num_agents)]
        for agent in range(num_agents):
            if collided_list[agent]:
                agent_states[agent] = "has collided!"
            elif breached_list[agent]:
                agent_states[agent] = "got lost!"
            elif reached_list[agent]:
                agent_states[agent] = "has reached!"
            else:
                agent_states[agent] = "is searching..."

        ## Write Doom text
        if self.doom:
            self._draw_text_doom()
            return

        ### Write normal game text
        ## Upper text
        # Game + Move info
        self.font.render_to(self.text_screen, (5, 5), "Game: " + str(game), self.white)
        self.font.render_to(
            self.text_screen,
            (5, 5 + self.text_spacing),
            "Move: " + str(move),
            self.white,
        )
        self.font.render_to(
            self.text_screen,
            (5, 5 + 2 * self.text_spacing),
            "Agent 1 " + agent_states[0],
            self.white,
        )
        self.font.render_to(
            self.text_screen,
            (5, 5 + 3 * self.text_spacing),
            "Agent 2 " + agent_states[1],
            self.white,
        )

        ## Lower text
        # Pause and Next Frame
        self.font.render_to(
            self.text_screen,
            (5, 0.5 * self.window_width - 2 * self.text_spacing),
            "Press p to pause/unpause",
            self.white,
        )
        self.font.render_to(
            self.text_screen,
            (5, 0.5 * self.window_width - self.text_spacing),
            "Press n to run next frame",
            self.white,
        )

    def _draw_text_doom(self):
        # Upper text
        self.font.render_to(self.text_screen, (5, 5), "Thank You", self.white)
        self.font.render_to(
            self.text_screen, (5, 5 + self.text_spacing), "Questions?", self.white
        )

    def _draw_to_true(self, display_info):
        """Draw agents and border to true_screen"""
        self._draw_border()
        self._draw_agents(display_info)
        self.true_screen.blit(
            self.border_screen, (self.env_size + 2 * self.border_size, 0)
        )
        self.true_screen.blit(
            self.env_screen, (self.env_size + 3 * self.border_size, self.border_size)
        )

    def _draw_to_scaled(self, display_info):
        """Draw true_screen and text to scaled screen"""
        self._draw_text(display_info)
        self.scaled_screen.blit(
            pygame.transform.scale(
                self.true_screen, self.scaled_screen.get_rect().size
            ),
            (0, 0),
        )
        self.scaled_screen.blit(self.text_screen, (0, 0))

    def _clear_screens(self):
        self.border_screen.fill("black")
        self.env_screen.fill("black")
        self.text_screen.fill("black")

    def _get_user_input(self):
        """
        Checks for screen quit, pause, or resize
        """
        self.next_frame = False
        # Check Events
        for event in pygame.event.get():
            # QUIT event
            if event.type == pygame.QUIT:
                self.headless = True
                pygame.display.quit()
            # PAUSE event
            elif event.type == pygame.KEYUP:
                if event.key == K_p:
                    self.pause = True
            # RESIZE event
            elif event.type == VIDEORESIZE:
                if (
                    min(event.size) > self.window_width
                ):  # limit max width to self.window_width
                    self.display_screen = pygame.display.set_mode(
                        (self.window_width, int(self.window_width * 0.5)),
                        DOUBLEBUF | RESIZABLE,
                    )
                else:
                    self.display_screen = pygame.display.set_mode(
                        (min(event.size) * 2, min(event.size)), DOUBLEBUF | RESIZABLE
                    )
        # WHEN PAUSED
        while self.pause and not self.next_frame:
            for event in pygame.event.get():
                if event.type == KEYUP:
                    if event.key == K_n:
                        # Run one frame
                        self.next_frame = True
                    if event.key == K_p:
                        # Unpause
                        self.pause = False

    def _draw_to_display(self):
        """
        Blits scaled_screen to display_screen
        Clears surfaces
        Flips display
        """
        if self.headless:
            return
        self.display_screen.blit(
            pygame.transform.scale(
                self.scaled_screen, self.display_screen.get_rect().size
            ),
            (0, 0),
        )
        self._clear_screens()
        pygame.display.flip()

    def _refresh_display(self, display_info):
        """
        Gets user input
        Draws border + env screens to true_screen
        Draws scaled true_screen and text_screen to scaled_screen
        Scales and draws scaled_screen to display_screen, flips display
        """
        self._get_user_input()
        self._draw_to_true(display_info)
        self._draw_to_scaled(display_info)
        self._draw_to_display()

    def display(self, display_info):
        if self.headless:
            return
        self._refresh_display(display_info)


class ScreenConfig:
    """
    Stores config information about pygame display
    """

    def __init__(
        self,
        headless=False,
        border_size=10,
        font_size=20,
        font="constantia",
        window_width=1500,
        text_spacing=20,
    ):
        self.headless = headless
        self.border_size = border_size
        self.font_size = font_size
        self.font = font
        self.window_width = window_width
        self.text_spacing = text_spacing


class DisplayInfo:
    """
    Stores information about the simulation that we would like to transfer to the Window object
    """

    def __init__(
        self,
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
        move,
    ):
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


####################################### main() ####################################


def main():
    gameconfig = GameConfig(
        env_size=256,
        config=1,
        speed=8,
        num_agents=2,
        agent_size=32,
        channels=4,
        num_actions=5,
    )
    screenconfig = ScreenConfig(
        headless=False,
        border_size=10,
        font_size=25,
        font="constantia",
        window_width=1000,
        text_spacing=25,
    )
    window = Window(screenconfig, gameconfig)

    i = 0
    while True:
        i += 1
        agent_pos = [
            [
                random.randint(0, gameconfig.env_size),
                random.randint(0, gameconfig.env_size),
            ],
            [
                random.randint(0, gameconfig.env_size),
                random.randint(0, gameconfig.env_size),
            ],
        ]
        target_pos = [
            [
                random.randint(0, gameconfig.env_size),
                random.randint(0, gameconfig.env_size),
            ],
            [
                random.randint(0, gameconfig.env_size),
                random.randint(0, gameconfig.env_size),
            ],
        ]

        display_info = DisplayInfo(
            agent_pos=agent_pos,
            target_pos=target_pos,
            action_list=[0, 0],
            reward_list=[0, 0],
            done_list=[0, 0],
            collided_list=[0, 0],
            reached_list=[0, 0],
            breached_list=[0, 0],
            done=1,
            game=1,
            move=0,
        )
        window.display(display_info=display_info)

        time.sleep(0.3)


if __name__ == "__main__":
    main()
