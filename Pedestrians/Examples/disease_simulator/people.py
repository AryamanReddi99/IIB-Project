import random
import numpy as np

STATUS = (S_healthy,
          S_infected,
          S_recovered,
          S_dead) = range(4)

BOUNDS = np.array((800, 800))

RED = (1, 0, 0)    # Infected
YELLOW = (1, 1, 0)  # Recovered
GREEN = (0, 1, 0)  # Healthy
BLUE = (0, 0, 1)   # 

QUARANTINE = True # False # 

stupidity_chance = 0
infection_chance = 0.1
immunity_loss_chance = 0.0001

class Person:
    def __init__(self, id: int, position: np.array, destination=BOUNDS/8):
        self.id = id
        self.status = S_healthy
        self.infection = None
        self.colour = GREEN
        self.position = position
        self.speed = 8
        self.destination_reached = True
        self.positive_test = False
        self.destination = destination
        self.stupid = random.random() < stupidity_chance

    def find_nearest_neighbour(self, i_people):
        """Checks for any infected people within range
        
        If so, roll for infection for each infected person
        until exhausted or until infected
        """
        for person in i_people:
            if person.id != self.id:
                distance = np.linalg.norm(person.position - self.position)
                if distance < 10:
                    if random.random() < infection_chance:
                        self.infect()
                        return person.id
        return -1


    def move(self, network):
        """Move the person

        If they're not at their destination, move them in the
        direction of the destination a small random amount,
        otherwise move randomly, and change destination if
        neccessary.
        """
        if (self.destination_reached and
                self.status != S_infected and
                random.random() < 0.003):
            self.change_destination(network)
        
        elif(self.positive_test and not self.stupid and
                not (self.destination==network[0]).all()):
            self.change_destination(network)
        elif(not self.positive_test and not self.stupid and
                (self.destination==network[0]).all()):
            self.change_destination(network)

        distance = np.linalg.norm(self.destination - self.position)

        if distance < 50:
            self.destination_reached = True
            for i in range(2):
                self.position[i] += random.randint(-self.speed/2, self.speed/2)
        else:
            direction = (self.destination - self.position)/distance

            self.position = self.position + direction*np.array((random.randint(-1,self.speed+1),
                                                                random.randint(-1,self.speed+1)))

        for i in range(2):
            if self.position[i] >= BOUNDS[i]:
                self.position[i] += random.randint(-self.speed, 0)
            elif self.position[i] < 0:
                self.position[i] = random.randint(0, self.speed)

    def change_destination(self, network):
        """Change current destination"""
        if QUARANTINE:
            if (self.positive_test and not self.stupid):
                self.destination = network[0]
                self.destination_reached = False
            elif self.stupid:
                pass
                self.destination = random.choice(network)
                self.destination_reached = False
            else:
                self.destination = random.choice(network[1:])
                self.destination_reached = False
        else:
                self.destination = random.choice(network[1:])
                self.destination_reached = False

    def fight_infection(self):
        """Develop infection"""
        if self.status == S_infected:
            infection_status = self.infection.change_severity()
            if infection_status == 1:
                self.cured()
            elif infection_status == -1:
                self.dead()
            return infection_status
        else: # should never happen now
            return 0

    def test_immunity(self):
        """If cured, roll to see if you lose immunity"""
        if random.random() < immunity_loss_chance:
            self.lose_immunity()
            return 1
        else:
            return 0

    def infect(self):
        #print(f"{self.id} is infected!")
        self.infection = Infection()
        self.status = S_infected
        if self.positive_test:
            self.colour = (1,0,1)
        else:
            self.colour = RED

    def cured(self):
        #print(f"{self.id} has recovered!")
        self.infection = None
        self.status = S_recovered
        if self.positive_test:
            self.colour = (1,0.5,0)
        else:
            self.colour = YELLOW

    def dead(self):
        """Obscelete"""
        #print(f"{self.id} has died!")
        self.infection = None
        self.status = S_dead
        self.colour = BLUE

    def lose_immunity(self):
        #print(f"{self.id} has lost immunity!")
        # No infection to gain or lose
        self.status = S_healthy
        if self.positive_test:
            self.colour = (0,1,1)
        else:
            self.colour = GREEN

    def take_test(self):
        """Simulate a test"""
        if self.status == S_healthy or self.status == S_recovered:
            if random.random() < 0.05:
                # False +ve
                self.positive_test = True
                if self.status == S_recovered:
                    self.colour = (1,0.5,0)
                else:
                    self.colour = (0,1,1)
            else:
                self.positive_test = False
                if self.status == S_recovered:
                    self.colour = YELLOW
                else:
                    self.colour = GREEN

        elif self.status == S_infected:
            if random.random() < 0.95:
                self.positive_test = True
                self.colour=(1,0,1)
            else:
                # False -ve
                self.positive_test = False
                self.colour = RED

    def next_day(self):
        """ Anything that needs to be done to begin the new day 
        
        (obscelete)
        """
        self.move()


class Infection:
    def __init__(self):
        #self.start_date = date
        self.severity = 0

    def change_severity(self):
        """ If severity goes above 5, the person dies
            if it goes below -5, they recover
        """
        self.severity += random.randint(-4, 5)
        if self.severity >= 200:
            return 1 # Recovered 
        elif self.severity <= -20:
            return -1 # Dead
        else:
            return 0
