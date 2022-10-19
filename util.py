from collections import namedtuple
from dataclasses import dataclass
import numpy as np 
Pos = namedtuple('Pos', ['x','y'])

@dataclass
class Obs:
    position: Pos

ACTIONS = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

STATUS_TO_CODE = {'Open':0,'Blocked':1,'Ghost':2,'Agent':'3'}
CODE_TO_STATUS = {v:k for (k,v) in STATUS_TO_CODE.items()}
CELL_PRINT_VAL = {0:' ',1:'█',2:'G',3:'Ag'}

CELL_PROBABILITY = 0.72   # DEFINED BY ASSIGNMENT.

def is_valid_agent_state(grid,position):
    size = len(grid)
    if position.x>=0 and position.x < size and position.y>=0 and position.y < size and grid[position.x,position.y]!=1:
        return True
    else:
        return False

def is_safe_agent_state(grid,position):
    size = len(grid)
    if position.x>=0 and position.x < size and position.y>=0 and position.y < size and grid[position.x,position.y]!=1 and grid[position.x,position.y]!=2:
        return True
    else:
        return False

def get_neighborhood_preference(grid,agent_location,size,range = 1):

    # Utility function to compute Agent5's neighborhood preference. 
    start = (max(0,agent_location.x-1), max(0,agent_location.y-1))
    stop = (min(size-1,agent_location.x+1), min(size-1,agent_location.y+1))
    distance_from_target_scale = (size - agent_location.x + size - agent_location.y) /(2*size)
    return np.sum(grid[start[0]:stop[0],start[1]:stop[1]])
    
