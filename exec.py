import imp
import itertools
from pickletools import optimize
from re import search
import numpy as np
from env import GridWorldEnv
from agents import Agent1, Agent2, Agent3, Agent4, Agent5
from collections import Counter
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm 
MULTI_THREADING=2

import nevergrad as ng
import pickle

def run_episode(env_size, ghosts, verbose, agent_config,blocking_walls):

    # function to run an episode.
    env = GridWorldEnv(env_size, ghosts,verbose,blocking_walls)

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
    return episode_success, episode_num

def run(number, env_size, ghosts, verbose,agent_config,blocking_walls):

    # wrapper compatible with Pool.map.
    run_result = run_episode(env_size,ghosts,verbose,agent_config,blocking_walls)
    
    return run_result


def run_brute_search():
    # Execute multi-processing jobs to collect agent performances for different number of ghosts and wall opacity configurations. 
    agent4_config = (Agent4,10,1)
    agent1_config = (Agent1,)
    agent2_config = (Agent2,2)
    agent3_config = (Agent3,2,2,5)
    agent4_config = (Agent4,10,50)

    results = []
    env_config = itertools.product([True,False], [5,10, 15, 20, 25, 30])
    #env_config = [(False,u) for u in [5,10,15,20,25,30, 40, 60, 80, 100]]

    for blocking_walls, ghosts in env_config:


        search_values =[(Agent4,50,0.5),agent1_config,agent2_config,agent3_config,(Agent5,50,0.5,1,1)]



        #stdscr= curses.initscr()
        for idx,search_val in enumerate(search_values):

            if MULTI_THREADING <=1:
                run_results = Counter()
                eps = []
                for i in tqdm(range(experiment_repeats)):
                    run_result = run_episode(env_size,ghosts,verbose,agent_config=search_val,blocking_walls=blocking_walls)
                    #print('Episode {0} success {1}'.format(i,run_result))
                    run_results.update({run_result[0]:1})
                    eps.append(run_result)
                    #stdscr.addstr(0,0,'Episode {0} success {1}'.format(i,run_result))
                    #stdscr.addstr(0,0,str(run_results))
                    #stdscr.refresh()
                results.append((blocking_walls,ghosts,search_val,run_results,np.mean([u[1] for u in eps if u>0]),
                                                                        np.std([u[1] for u in eps if u>0]),eps))
            else:
                with Pool(MULTI_THREADING) as pool:
                    run_results = pool.map(partial(run,env_size = env_size,ghosts=ghosts,verbose=verbose,
                                                        agent_config=search_val,blocking_walls=blocking_walls),list(range(experiment_repeats)))
                    results.append((blocking_walls,ghosts,search_val,Counter([u[0] for u in run_results]),np.mean([u[1] for u in run_results if u[1]>0]),
                                                                                                                    np.std([u[1] for u in run_results if u[1]>0]),run_results))
            print('{1}/{2} Episode success rate {0}'.format(results[-1][:-1],idx,len(search_values)))
    print('\n')
    #print(results)
    print('\n'.join(str(u) for u in results[:-1]))
    with open('out_agent5_true_1.pkl','wb') as fout:
        pickle.dump(results,fout)

def run_agent4_grad_free_optim():

    # Run a gradient free optimization technique to find the parameters of Agent4. 
    
    def build_results(penalty_multiplier, penalty_exponent):
        search_val = (Agent4,penalty_multiplier,penalty_exponent)
        with Pool(MULTI_THREADING) as pool:
            run_results = pool.map(partial(run,env_size = env_size,ghosts=ghosts,verbose=verbose,
                                                agent_config=search_val,blocking_walls=blocking_walls),list(range(experiment_repeats)))
            res_dict = Counter([u[0] for u in run_results]) 
            return -res_dict['Win']/len(run_results)
    
    parametrization = ng.p.Instrumentation(
    penalty_multiplier=ng.p.Log(lower=.01, upper=20.), 
    penalty_exponent= ng.p.Log(lower=0.001, upper=10.0))

    optimizer = ng.optimizers.NGOpt(parametrization = parametrization, budget=20)
    recommendation = optimizer.minimize(build_results,verbosity=2)

    print(recommendation.kwargs)



if __name__=="__main__":
    env_size = 51
    ghosts = 15
    experiment_repeats = 1000
    verbose = False
    blocking_walls = True
    run_agent4_grad_free_optim()
    run_brute_search()
