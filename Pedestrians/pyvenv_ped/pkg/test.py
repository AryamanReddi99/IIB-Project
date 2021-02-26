import pygame
import os
from pygame.locals import *
from pathlib import Path

def main():
    l1 = (1,2,3,4,5)
    l2 = (1,44,5,3,5)
    l3 = (1,2,3,4,5)
    l4 = (1,44,5,3,5)
    l5 = (1,2,3,4,5)
    l6 = (1,44,5,3,5)

    l_all = [l1,l2,l3,l4,l5,l6]

    print(zip(*l_all))
    a,b,c,d,e = zip(*l_all)
    print(a)
    

main()
