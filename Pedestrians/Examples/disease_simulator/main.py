import pygame
import random
import numpy as np

from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

from people import Person
from general_window import *


def next_day(h_people, i_people, r_people, network):
    """All actions people do in one frame
    
    Order of actions:
    1) Infected:
        i)   Infected people roll for infection development
        ii)  If they die/recover, move them to that list (or remove if dead)
        iii) Move each infected person
    2) Recovered / Cured:
        i)   Move each cured person
    3) Healthy:
        i)   Move person
        ii)  Roll for infection
    4) Everyone (exc. dead):
        i)   Sample for who gets a test
    """

    num_people = len(h_people + i_people + r_people)
    for person in i_people:
        status = person.fight_infection()
        if status == -1:
            i_people.remove(person)
        elif status == 1:
            r_people.append(person)
            i_people.remove(person)
            person.move(network)
        else:
            person.move(network)

    for person in r_people:
        person.move(network)
        if person.test_immunity() == 1:
            h_people.append(person)
            r_people.remove(person)

    for person in h_people:
        person.move(network)
        vector = person.find_nearest_neighbour(i_people)
        if vector != -1:
            i_people.append(person)
            h_people.remove(person)
    
    # Asymptomatic testing
    for person in random.sample(h_people + i_people + r_people,int(num_people/50)):
        person.take_test()

def draw_people(window, people):
    """Draw coloured dots to represent people"""

    params = ([], 4)
    for person in people:
        params[0].append((person.position, person.colour))
    #print(params)
    window.draw_points(*params)

def draw_network(window, network):
    """Draw dots to represent network"""
    params = ([], 100)
    params[0].append((network[0], (0.5,0,1)))
    for location in network[1:]:
        params[0].append((location, (0,0,1)))
    window.draw_points(*params)

def main():
    """Run the simulation

    Order:
    1) Set up window
    2) Set up adjustable parameters
    3) Initialise variables
    4) Infect the first person
    5) Main loop:
        i)   Store list lengths from previous loop
        ii)  Execute next_day()
        iii) Store current list lengths
        iv)  Print lengths if different to previous day
        v)   Clear window and draw objects
        vi)  Check if simulation has finished
    """

    bounds = (np.array((800, 800)))  # to match default in screenConfig
    screenConfig = ScreenConfigStruct(framerate=1)
    window = Window(screenConfig)



    ### PARAMETERS ###
    num_people = 200
    net_size = 10 # Must be >= 42
    ### ---------- ###



    h_people = [None]*num_people
    i_people = []
    r_people = []

    # Create vectors for all possible network points to sample from
    possible_locations = [None]*42 # Number of allowable locations for random placement
    for i in range(7):
        for j in range(6):
            possible_locations[6*i + j] = bounds*np.array(((i+1)/8, (j+1)/8))

    # Create network by sampling from possible points
    network = [bounds*np.array((4/8, 7/8))] + random.sample(possible_locations, net_size-1)

    # Create people, randomly assigning start locations
    for i in range(num_people):
        destination = random.choice(network[1:])
        position = destination + np.array(
            (random.randint(-50, 50), random.randint(-50, 50)))
        h_people[i] = Person(i, position, destination)

    # Infect the first person
    h_people[0].infect()
    i_people.append(h_people[0])
    h_people.remove(h_people[0])

    # Store current amounts
    healthy_now = len(h_people)
    infected_now = len(i_people)
    cured_now = len(r_people)

    # Begin loop checking if the window has been closed
    print("| Healthy |Infected |   Cured |    Dead |      log |")
    while window.check_quit() == 0:

        # Store previous loop's list lengths
        healthy_start = healthy_now
        infected_start = infected_now
        cured_start = cured_now


        # Execute a day
        next_day(h_people, i_people, r_people, network)

        # Get stats
        healthy_now = len(h_people)
        infected_now = len(i_people)
        cured_now = len(r_people)
        dead_now = num_people - healthy_now - infected_now - cured_now

        # Check if values have changed
        if (healthy_start != healthy_now or
                infected_start != infected_now or
                cured_start != cured_now):

            # Only print log if it's not -inf
            if healthy_now*infected_now > 0:
                print(f"|     {healthy_now:3d} "
                        f"|     {infected_now:3d} "
                        f"|     {cured_now:3d} |     {dead_now:3d} "
                        f"|     {np.log10(healthy_now*infected_now):4.1f} |")
            else:
                print(f"|     {healthy_now:3d} "
                        f"|     {infected_now:3d} "
                        f"|     {cured_now:3d} |     {dead_now:3d} "
                        f"|     ---- |")

        # Clear screen
        glClear(GL_COLOR_BUFFER_BIT)

        # Draw here
        draw_network(window, network)
        draw_people(window, h_people + i_people + r_people)

        # refresh
        window.refresh_display()

        # Check if complete (warning: exit button won't work)
        # if this gets triggered)
        if infected_now == 0:
            print(f"Infection over!\n"
                    f"\tHealthy: {healthy_now}\n"
                    f"\t  Cured: {cured_now}\n"
                    f"\t   Dead: {dead_now}")
            # Wait for terminal input to end program
            input()
            break


if __name__ == "__main__":
    main()