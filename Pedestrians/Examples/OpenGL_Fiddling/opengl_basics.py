# Simple pygame program

# Import and initialize the pygame library
import pygame
import random
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
pygame.init()
screen = pygame.display.set_mode([256, 256])

# Set up the drawing window

# Example points
blue = (0, 0, 255)
red  = (255, 0, 0)

agent_1 = [100,200]
agent_2 = [200,100]

# Run until the user asks to quit
running = True
while True:

    # Did the user click the window close button?


    # Fill the background with black
    screen.fill((0, 0, 0))

    # Draw a solid blue circle in the center
    agent_1 = [random.randint(0,256),random.randint(0,256)]
    agent_2 = [random.randint(0,256),random.randint(0,256)]

    pygame.draw.circle(screen, blue, agent_1, 5)
    pygame.draw.circle(screen, red, agent_2, 5)

    # Flip the display
    pygame.display.flip()

# Done! Time to quit.
pygame.quit()