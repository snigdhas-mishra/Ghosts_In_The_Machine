# Built using https://www.gymlibrary.dev/content/environment_creation/. 
# Can be modified to be registered to OpenAI gym later.

from tkinter.tix import CELL
import numpy as np
import pandas as pd
from collections import namedtuple, defaultdict
import random
import copy 
from math import floor
from util import *

DIRECTIONAL_VISIBILITY= True

class GridWorldEnv:

    def __init__(self,size, ghosts = 5,verbose= False,blocking_walls=False):
        self.verbose = verbose 
        self.size = size  # The size of the square grid

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).


        # We have 4 actions, corresponding to "right", "up", "left", "down"

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self.num_ghosts = ghosts
        self.allocate_resources()
        self.blocking_walls = blocking_walls

    def save_state(self):
        saved_death = copy.copy(self.death)
        saved_grid = np.array(self.grid,copy=True)
        saved_play_grid= np.array(self.play_grid,copy=True)
        saved_agent_path = np.array(self.agent_path, copy=True)
        saved_agent_location = copy.deepcopy(self.agent_location)
        saved_ghost_locations = copy.deepcopy(self.ghost_locations)

        return (saved_death,saved_grid,saved_play_grid,saved_agent_path,saved_agent_location,saved_ghost_locations)

    
    def load_state(self, state):
        self.death = copy.copy(state[0])
        self.grid = np.array(state[1],copy=True)
        self.play_grid= np.array(state[2],copy=True)
        self.agent_path = np.array(state[3], copy=True)
        self.agent_location = copy.deepcopy(state[4])
        self.ghost_locations = copy.deepcopy(state[5])

        return 


    def allocate_resources(self):
        self.target = Pos(self.size-1,self.size-1)
        self.source = Pos(0,0)
        self.death = False
        while True:   
            self.grid = (np.random.random((self.size,self.size)) > CELL_PROBABILITY).astype('float')
            self.play_grid = np.array(self.grid,copy= True)
            self.grid[0,0]=0.
            self.grid[self.target.x,self.target.y]=0.
            self.grid[-1,-1]=0. 
            self.print_grid()

            if self.solvable():
                if self.verbose:
                    print('The grid is solvable. Setting up the environment ...')
                self.precompute_shortest_paths()
                break
            else:
                if self.verbose:
                    print('The grid is not solvable. Resetting the environment ...')
            
        self.let_there_be_ghosts()
        self.agent_path = np.zeros((self.size,self.size)) -1.0 # All the cells  that the agents has never visited are -1
        self.init_agent()
        if self.verbose:
            print('Playboard with ghosts and agent')
            self.print_grid()
            print('Game initialization complete')

    def get_init_data(self):
        return (self.sh_path, self.sh_path_dist, self.grid)

    def obs(self):
        return self.play_grid, self.agent_location

    def step_agent(self,direction):
        new_X,new_Y = (self.agent_location.x+direction[0], self.agent_location.y + direction[1])
        step_success = is_valid_agent_state(self.grid,Pos(new_X,new_Y))
        if step_success:
            previous_location = self.agent_location
            self.agent_location= Pos(new_X,new_Y)
            self.play_grid[previous_location.x,previous_location.y]=0
            self.play_grid[new_X,new_Y]=3
            self.agent_path[new_X,new_Y]=1 + self.agent_path[previous_location.x,previous_location.y]
            if any(g==self.agent_location for g in self.ghost_locations):
                self.death = True
        else:
            print('Illegal Move.')
        return step_success

    def step_ghosts(self):
        for i in range(len(self.ghost_locations)):  
            gh = self.ghost_locations[i]
            possible_actions = []
            if gh.x!=self.size-1:
                possible_actions.append(0)
            if gh.x!=0:
                possible_actions.append(2)
            if gh.y!=self.size -1: 
                possible_actions.append(1)
            if gh.y!=0:
                possible_actions.append(3)
            selected_action = random.choice(possible_actions)
            potential_new_position = Pos(gh.x + ACTIONS[selected_action][0],gh.y + ACTIONS[selected_action][1])

            if self.grid[potential_new_position.x,potential_new_position.y]==1:
                if np.random.random() >0.5:
                    continue
            self.play_grid[gh.x,gh.y]= self.grid[gh.x,gh.y]
            self.play_grid[potential_new_position.x,potential_new_position.y]=2
            self.ghost_locations[i]=potential_new_position
        
    def step(self,direction):
        if self.death:
            if self.verbose:
                print('AGENT IS DEAD!!')
        step_success = self.step_agent(direction) 
        self.step_ghosts()
        if any(g==self.agent_location for g in self.ghost_locations):
            self.death = True
        return self.play_grid, step_success, self.death, self.agent_location, (self.agent_location == self.target)

    def init_agent(self):
        if self.verbose:
            print('Initialising agent ... ')
        self.agent_location= Pos(0,0)
        self.play_grid[0,0]=3
        self.agent_path[0,0]=1
    
    def let_there_be_ghosts(self):
        if self.verbose:
            print('Creating ghosts ...')
        #Pull all the nodes that can be visited from the target. 
        #Since there is a path from the source to target, all nodes visitable from the target can also be visited from the source.
        available_nodes = [node for node in self.sh_path if node!=self.source]
        random.shuffle(available_nodes)
        self.ghost_locations = available_nodes[:self.num_ghosts]
        for gh in self.ghost_locations:
            self.play_grid[gh.x,gh.y]=2

    def get_ghost_locations(self):
        if self.blocking_walls is False:
            return self.ghost_locations
        else:
            visible_ghosts=[]
            for ghost in self.ghost_locations:
                x_range = (min(ghost.x,self.agent_location.x),max(ghost.x,self.agent_location.x)+0.01)
                y_range = (min(ghost.y,self.agent_location.y),max(ghost.y,self.agent_location.y)+0.01)
                if x_range[0]==x_range[1]:
                    visibility = all([self.grid[x_range[0],y]!=1 for y in range(*y_range)])
                elif y_range[0]==y_range[1]:
                    visibility= all([self.grid[x,y_range[0]]!=1 for x in range(*x_range)])
                else:
                    diff = max(x_range[1]-x_range[0],y_range[1]-y_range[0])
                    x_steps= np.arange(x_range[0],x_range[1],(x_range[1]-x_range[0])/diff)
                    y_steps= np.arange(y_range[0],y_range[1],(y_range[1]-y_range[0])/diff)
                    #visibility = [self.grid[int(x),int(y)]!=1 for (x,y) in zip(x_steps,y_steps)]
                    #visibility = all(visibility)
                    visibility = (self.grid[x_steps.astype(np.int32),y_steps.astype(np.int32)]!=1).all()
                    #assert visibility1==visibility,'Error'
                if visibility is True:
                    visible_ghosts.append(ghost)
        return visible_ghosts


                

    def precompute_shortest_paths(self):
        if self.verbose:
            print('Precomputing shortest paths ... ')
        sh_path={}
        sh_path_dist = defaultdict(lambda : (self.size **2)*2)
        start_node=self.target
        visited_nodes={}
        queue = []
        queue.append(start_node)
        sh_path_dist[start_node]=0
        sh_path[start_node]=None

        while queue: 
            visiting = queue.pop(0)
            if visiting not in visited_nodes:
                valid_next_nodes = [_[0] for _ in self.get_valid_next_steps(visiting) if _[1] ==0 and _[0] not in visited_nodes]
                for node in valid_next_nodes:
                    if node in sh_path_dist:
                        #print(sh_path_dist[node],sh_path_dist[visiting]+1)
                        assert sh_path_dist[node]<=sh_path_dist[visiting]+1,"Node visitation error"
                        if sh_path_dist[node]==sh_path_dist[visiting]+1:
                            sh_path[node].append(visiting)
                    else:
                        sh_path[node]=[visiting,]
                        sh_path_dist[node]=sh_path_dist[visiting]+1
                        queue.append(node)
                visited_nodes[visiting]=True
        
        self.sh_path_dist = sh_path_dist
        self.sh_path = sh_path
        if self.verbose:
            print('Precomputation of shortest path to target .... done.')
            print('The shortest path for the agent (Ag) to target is {0}'.format(self.sh_path_dist[self.source]))
            print(' Printing the shortest path values for debugging ...')

            #Printing 
            shortest_path_rep =[[' ' for i in range(self.size)] for j in range(self.size)]
            for j in range(self.size):
                for i in range(self.size):
                    if self.grid[i,j]!=0:
                        shortest_path_rep[i][j]='X'
                    elif Pos(i,j) in sh_path and sh_path[Pos(i,j)] is not None:
                        #shortest_path_rep[i][j]=str((sh_path_dist[Pos(i,j)],len(sh_path[Pos(i,j)])))
                        shortest_path_rep[i][j]=str(sh_path_dist[Pos(i,j)])
                    else:
                        shortest_path_rep[i][j]=' '
            #shortest_path_rep = [[str((sh_path_dist[Pos(i,j)],len(sh_path[Pos(i,j)]))) if self.grid[i,j]==0 and Pos(i,j) in sh_path
            #                                             else 'X' for i in range(self.size)] for j in range(self.size)]
            print(pd.DataFrame(shortest_path_rep))
        return 
    
        

    def get_valid_next_steps(self,pos):
        valid_actions=[]
        for _,direction in ACTIONS.items():
            new_X,new_Y = (pos.x+direction[0], pos.y + direction[1])
            if new_X>=0 and new_X < self.size and new_Y>=0 and new_Y < self.size:
                valid_actions.append((Pos(new_X,new_Y),self.grid[new_X,new_Y]))

        return valid_actions

    def print_grid(self):
        if self.verbose:
            printable_vals = [[CELL_PRINT_VAL[u] for u in v] for v in self.play_grid]
            print(pd.DataFrame(printable_vals))

    def print_grid_with_agent_path(self):
        printable_vals = [[CELL_PRINT_VAL[u] for u in v] for v in self.play_grid]
        for x in range(self.size):
            for y in range(self.size):
                if self.agent_path[x,y]!=-1:
                    printable_vals[x][y] = str(int(self.agent_path[x,y]))
        print(pd.DataFrame(printable_vals))

    def quick_dfs(self):
        start_node=Pos(0,0)
        visited_nodes={}
        queue = []
        queue.append(start_node)

        while queue:
            visiting = queue.pop(0)
            if visiting not in visited_nodes:
                valid_next_nodes = [_[0] for _ in self.get_valid_next_steps(visiting) if _[1] ==0 and _[0] not in visited_nodes]
                if any([_==self.target for _ in valid_next_nodes]):
                    return True
                queue = valid_next_nodes + queue
                visited_nodes[visiting]=True

        return False
    


    def solvable(self):
        return self.quick_dfs()



if __name__=="__main__":
    #Testing the printing 
    u = GridWorldEnv(10)
    u.step(ACTIONS[0])
    u.print_grid()
