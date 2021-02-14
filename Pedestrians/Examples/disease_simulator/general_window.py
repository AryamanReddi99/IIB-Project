import pygame
import random
import numpy as np

from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

class Window:
    """Handles the pygame window"""
    def __init__(self, screenConfig):
        """initialise parameters

        Currently only works for orthogonal view, but this will
        change eventually.
        """
        print("initialising window")
        self.clock = pygame.time.Clock()
        self.framerate = screenConfig.framerate

        pygame.init()
        windowSize = screenConfig.windowSize
        xBounds = screenConfig.xBounds 
        yBounds = screenConfig.yBounds
        pygame.display.set_mode(windowSize, flags=DOUBLEBUF | OPENGL) # dont mess with this

        if screenConfig.perspective == "ortho": # infinitely far away
            glOrtho(*xBounds, *yBounds,-1, 1)
        else:
            zBounds = screenConfig.zBounds

    def check_quit(self):
        """Returns 1 if the close button on the window is pressed"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 1
        return 0

    def draw_lines(self, lines, colour=(0,1,0)):
        """Draws a set of lines with a given colour
        
        Lines are pairs of 2d/3d coordinates with
        a colour tuple at the end, OR a colour tuple
        given seperately, as the colour of all given
        lines.
        """
        
        # Find dimension of first line's start coords
        dimension = len(lines[0][0])

        # Check if there's a colour vector
        ifColour = (len(lines[0]) == 3)

        glBegin(GL_LINES)
        if not ifColour:
            glColor3d(*colour)

        if dimension == 2:
            vertexFunc = glVertex2f
        elif dimension == 3:
            vertexFunc = glVertex3f
        else:
            raise Exception("Dimension must be 2 or 3")

        for line in lines:
            if ifColour:
                glColor3d(*line[2])
            vertexFunc(*line[0])
            vertexFunc(*line[1])

        glEnd()

    def draw_points(self, points, size=2, colour=(0,1,0)):
        """Draws a set of points with a given size and colour

        Points are a set of 2D or 3D coordinates with a size int
        and/or a colour vector on the end, OR size and colour given
        seperately, as the size and colour of all given points.
        WARNING: for some reason given a general colour doesn't work
        """
        # Find dimension of first point's coords
        dimension = len(points[0][0])
        num_attribs = len(points[0])
        attribs = ""

        vertexFunc = None
        #attribFuncs = None

        glEnable(GL_POINT_SMOOTH)

        if num_attribs == 1:
            glPointSize(size)
            glColor3d(*colour) # color3d = RBG color
            glBegin(GL_POINTS)
            # attribFuncs is default

        elif num_attribs == 2:
            if len(points[0][1]) == 1: #size given, not colour
                glBegin(GL_POINTS)
                glColor3d(*colour)
                attribs = "size"
            elif len(points[0][1]) == 3: #colour given, not size
                glPointSize(size)
                glBegin(GL_POINTS)
                attribs = "colour"
            else:
                raise Exception("Unknown argument in point array")
        elif num_attribs == 3:
            attribs = "size colour"
            glBegin(GL_POINTS)
        else:
            raise Exception(f"Too many arguments in point array ({num_attribs})")

        # Change which function is used depending on dimension
        if dimension ==2:
            vertexFunc = glVertex2f
        elif dimension == 3:
            vertexFunc = glVertex3f
        
        if attribs == ("size"):
            for point in points:
                glPointSize(point[1])
                vertexFunc(*point[0])
        elif attribs == ("colour"):
            for point in points:
                glColor3d(*point[1])
                vertexFunc(*point[0])
        elif attribs == "size colour":
            for point in points:
                glPointSize(point[1])
                glColor3d(*point[2])
                vertexFunc(*point[0])
        glEnd()

    def clear_window(self):
        """Removes all drawings from screen"""
        glClear(GL_COLOR_BUFFER_BIT) # clear buffer on easel
    
    def refresh_display(self):
        """Refreshes screen"""
        pygame.display.flip() # flip the 2 buffers (visible one and the one being drawn)
        if self.framerate != None:
            self.clock.tick(30)


class ScreenConfigStruct:
    def __init__(self, windowSize=(800,800),
                       xBounds=(0,800),
                       yBounds=(0,800),
                       zBounds=None,
                       perspective="ortho",
                       framerate=30):

        self.windowSize = windowSize
        self.xBounds = xBounds
        self.yBounds = yBounds
        self.zBounds = zBounds
        self.perspective = perspective
        self.framerate = framerate