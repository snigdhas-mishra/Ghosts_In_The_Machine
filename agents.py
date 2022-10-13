from platform import node
import numpy as np

from util import * 
class Agent:
    def __init__(self,env) -> None:
        self.env = env
        self.sh_path, self.sh_path_dist, self.grid = self.env.get_init_data()


class Agent1(Agent):

    def __init__(self, env) -> None:
        super().__init__(env)
    
    def next_action(self,agent_location):
        self.agent_location = agent_location
        next_state = self.sh_path[self.agent_location][0]
        next_action = np.array([next_state.x - self.agent_location.x,next_state.y-self.agent_location.y])
        return next_action

class Agent2(Agent):
    def __init__(self, env,threshold) -> None:
        super().__init__(env)
        self.play_grid, self.agent_location = self.env.obs()
        self.size =self.env.size
        self.threshold = threshold
    
    def next_action(self,agent_location):
        self.agent_location = agent_location
        limited_bfs_result = self.run_limited_bfs()
        if limited_bfs_result is not None:
            return limited_bfs_result
        else:
            min_ghost = self.get_nearest_ghost()
            if min_ghost is None:
                return None
            x_action = (int(np.sign(self.agent_location.x - min_ghost.x)),0)
            y_action = (0,int(np.sign(self.agent_location.y - min_ghost.y)))

            result_action=[]
            if x_action!=(0,0):
                result_action.append(x_action)
            if y_action!=(0,0):
                result_action.append(y_action)

            for action in result_action:
                print(action)
                if is_safe_agent_state(self.play_grid,Pos(agent_location.x+action[0],agent_location.y+action[1])):
                    return action
            #print(min_ghost,self.agent_location,action)
            return (0,0)


    def get_nearest_ghost(self):
        action = [0,0]
        ghost_locations = self.env.get_ghost_locations()
        if len(ghost_locations)==0:
            return None
        min_dist=self.size**2
        min_ghost = ghost_locations[0]
        for g in ghost_locations:
            dist = np.abs(g.x - self.agent_location.x) + np.abs(g.y - self.agent_location.y)
            if dist < min_dist:
                min_ghost = g
        
        return g



    def run_limited_bfs(self):
        nodes_at_threshold =[]
        visited={}
        next_action_locations = [(u,1,Pos(self.agent_location.x + u[0],self.agent_location.y +u[1])) for u in ACTIONS.values()]
        for ne in next_action_locations:
            visited[ne[2]]=True
        queue = [u for u in next_action_locations if is_safe_agent_state(self.play_grid,u[2])]
        while len(queue)>0:
            act,count,loc = queue.pop(0)
            if count>=self.threshold:
                nodes_at_threshold.append((act,count,loc))
                continue
            next_locations = [(act,count+1,Pos(loc.x + u[0],loc.y +u[1])) for u in ACTIONS.values()]
            next_locations = [v for v in next_locations if is_safe_agent_state(self.play_grid,v[2]) if v[2] not in visited]
            queue = queue + next_locations
            for ne in next_locations:
                visited[ne[2]]=True
        
        if len(nodes_at_threshold)!=0:
            nodes_at_threshold.sort(key=lambda x:self.sh_path_dist[x[2]])
            return nodes_at_threshold[0][0]
        


        
class Agent3(Agent):
    def __init__(self, env,threshold, simulation_threshold, num_simulations =5) -> None:
        super().__init__(env)
        self.play_grid, self.agent_location = self.env.obs()
        self.agent2 = Agent2(env,threshold=threshold)
        self.env = env 
        self.threshold = threshold
        self.simulation_threshold = simulation_threshold
        self.num_simulations = num_simulations


    def next_action(self,agent_location):
        self.agent_location = agent_location
        next_action_locations = [(u,Pos(self.agent_location.x + u[0],self.agent_location.y +u[1])) for u in ACTIONS.values()]
        next_action_locations = [u for u in next_action_locations if is_safe_agent_state(self.play_grid,u[1])]
        saved_game_state = self.env.save_state()
        results =[]
        for next_action in next_action_locations:
            death_probability, mean_remaining_path = self.simulate_future(next_action[0],saved_game_state)
            self.env.load_state(saved_game_state)
            self.agent_location = self.env.obs()[1]
            results.append((death_probability, mean_remaining_path,next_action[0]))
        
        if len(results)>0:
            results.sort(key=lambda x:x[:2])
            next_action = results[0][-1]
        else:
            next_action = self.agent2.next_action(self.agent_location)

        return next_action

    def simulate_future(self,next_action,saved_game_state):
        death_probability = []
        remaining_path = []
        initial_action = next_action
        for idx in range(self.num_simulations):
            next_action = initial_action
            self.env.load_state(saved_game_state)
            simulation_len = self.simulation_threshold
            play_grid, step_success, death, agent_location, success = self.env.step(next_action)

            while death is False and success is False and simulation_len>0:
                #print('Episode No {0}/{1}'.format(episode_num,episode_limit))
                #episode_num +=1
                next_action = self.agent2.next_action(agent_location)
                play_grid, step_success, death, agent_location, success = self.env.step(next_action)
                if step_success: 
                    simulation_len-=1
            if death:
                death_probability.append(1)
            else:
                death_probability.append(0)
                remaining_path.append(self.env.sh_path_dist[agent_location])

        
        return (1. if len(remaining_path)==0 else np.mean(death_probability), +10000. if len(remaining_path)==0 else np.mean(remaining_path))


class Agent4(Agent):
    def __init__(self, env, penalty_multiplier, penalty_exponent) -> None:
        super().__init__(env)
        self.play_grid, self.agent_location = self.env.obs()
        self.size =self.env.size
        self.ghost_penalty_threshold = 5
        self.penalty_multiplier =penalty_multiplier
        self.penalty_exponent = penalty_exponent
        self.penalty_func = lambda x: np.exp(-x/penalty_exponent)

    def next_action(self,agent_location):
        self.agent_location = agent_location
        next_action_locations = [(u,Pos(self.agent_location.x + u[0],self.agent_location.y +u[1])) for u in ACTIONS.values()]
        next_action_locations = [u for u in next_action_locations if is_safe_agent_state(self.play_grid,u[1])]
        if len(next_action_locations)==0:
            return (0,0)
        next_action_sh_path = [self.env.sh_path_dist[u[1]] for u in next_action_locations]
        max_sh_path = max(next_action_sh_path)
        max_sh_path = max(1,(max_sh_path*self.penalty_multiplier))
        #max_sh_path = self.penalty_exponent
        next_action_sh_pen = [(sum(self.get_ghost_penalty_heuristic(next_action_locations[idx][1]))* max_sh_path + next_action_sh_path[idx]
                                                                                    ,next_action_sh_path[idx],*next_action_locations[idx]) 
                                                                                    for idx in range(len(next_action_locations))]
        next_action_sh_pen.sort(key= lambda x:x[:2])
        return next_action_sh_pen[0][2]

    def get_ghost_penalty_heuristic(self,loc):
        penalty = []
        for ghost in self.env.get_ghost_locations():
            x_diff = np.abs(ghost.x-loc.x)
            y_diff = np.abs(ghost.y-loc.y)
            if x_diff<self.ghost_penalty_threshold or y_diff <self.ghost_penalty_threshold:
                penalty.append((self.penalty_func(x_diff+y_diff)))
        return penalty