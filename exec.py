import imp
import itertools
from re import search
import numpy as np
from env import GridWorldEnv
from agents import Agent1, Agent2, Agent3, Agent4
from collections import Counter
from multiprocessing import Pool
from functools import partial
import curses
MULTI_THREADING=4

def run_episode(env_size, ghosts, verbose, agent_config,blocking_walls):
    env = GridWorldEnv(env_size, ghosts,verbose,blocking_walls)
    #agent = Agent1(env)
    #agent = Agent2(env,2)
    #agent = Agent3(env,2,2,5)
    agent = agent_config[0](env,*agent_config[1:])
    death = False
    episode_limit = 5000
    episode_success ='Unresolved'
    episode_num =0
    play_grid, agent_location = env.obs()
    success = False
    while death is False and success is False and episode_limit>episode_num:
        #print('Episode No {0}/{1}'.format(episode_num,episode_limit))
        episode_num +=1
        next_action = agent.next_action(agent_location)
        play_grid, step_success, death, agent_location, success = env.step(next_action)

        #env.print_grid()
    #env.print_grid_with_agent_path()
    if success:
        episode_success='Win'
    else: 
        if death:
            episode_success='Dead'
        if episode_num>=episode_limit:
            episode_success='Timeout'
    return episode_success, episode_limit - episode_num

def run(number, env_size, ghosts, verbose,agent_config,blocking_walls):
    run_result = run_episode(env_size,ghosts,verbose,agent_config,blocking_walls)
    
    #print('Episode {0} success {1}'.format(number,run_result))
    return run_result

if __name__=="__main__":
    agent4_config = (Agent4,10,1)
    agent1_config = (Agent1,)
    agent2_config = (Agent2,2)
    agent3_config = (Agent3,2,2,5)
    agent4_config = (Agent4,10,50)
    env_size = 51
    ghosts = 5
    experiment_repeats = 1000
    verbose = False
    blocking_walls = True
    results = []
    for blocking_walls in [True,False]:

        search_values = itertools.product([20,10,5,2,1,0.5,0.1],[2,1,0.5,0.1,0.05,0.01])
        #search_values = itertools.product([2,1],[1,0.5])
        search_values = [(Agent4,*u) for u in search_values]
        search_values = search_values + [agent1_config,agent2_config,agent3_config]
        #search_values =[(Agent4,50,0.5),agent1_config,agent2_config,agent3_config]
        #search_values =[(Agent4,20,1),agent1_config]

        #stdscr= curses.initscr()
        for idx,search_val in enumerate(search_values):

            if MULTI_THREADING <=1:
                run_results = Counter()
                eps = []
                for i in range(experiment_repeats):
                    run_result = run_episode(env_size,ghosts,verbose,agent_config=search_val,blocking_walls=blocking_walls)
                    #print('Episode {0} success {1}'.format(i,run_result))
                    run_results.update({run_result[0]:1})
                    #stdscr.addstr(0,0,'Episode {0} success {1}'.format(i,run_result))
                    #stdscr.addstr(0,0,str(run_results))
                    #stdscr.refresh()
                results.append((blocking_walls,search_val,run_results))
            else:
                with Pool(MULTI_THREADING) as pool:
                    run_results = pool.map(partial(run,env_size = env_size,ghosts=ghosts,verbose=verbose,
                                                        agent_config=search_val,blocking_walls=blocking_walls),list(range(experiment_repeats)))
                    print(run_results[:2])
                    results.append((blocking_walls,search_val,Counter([u[0] for u in run_results]),np.mean([u[1] for u in run_results if u[1]>0]),
                                                                                                                    np.std([u[1] for u in run_results if u[1]>0])))
            print('{1}/{2} Episode success rate {0}'.format(results[-1],idx,len(search_values)))
    print('\n')
    #print(results)
    print('\n'.join(str(u) for u in results))