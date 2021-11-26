# -*- coding: utf-8 -*-
"""
@author:  Pin-Han, Huang
"""
# In[0]:
import pandas as pd
import numpy as np
import os
from config import params, method_name
from methods import (hill_climbing_dist, random_walk_dist, simu_anneal_dist, 
                     tabu_search_dist, PSO_dist, GA, run_ant_colony, 
                     tabu_search_second, Hybrid_PSO_TS)
# In[1]:
#main_function 
def run_expr(method_list, iteration, PSO_iter, GA_iter, ant_iter, first_stg, second_stg):
    path = './'   
    data = pd.read_csv(path+'Distance.csv', header = 0, index_col=0)
    distance = data.T
    
    #get dataframe values as matrix and save it for plot
    distance_matrix = distance.values
    #set the stored path
    stored_path = '...stored_path_here...' 
    np.save(os.path.join(stored_path, 'dist_matrix.npy'), distance_matrix)  
    
    #create dataframe for distance abd save it for plot
    dist_table = pd.DataFrame(distance_matrix, columns = range(0,len(distance_matrix))) #(0,36) 
     
    # create destination
    destin  = list(distance.columns)
    #mapping index to desti
    index2desti = dict()
    for i in range(len(destin)):
        index2desti[i] = destin[i]
    
    #run the experiments
    #hill_climbing
    hc_best_dist, hc_path = hill_climbing_dist(dist_table, iteration)
    np.savez(os.path.join(stored_path,'hc_results.npz'), dist = hc_best_dist, path = hc_path)
    #hill_climbing results
    path_hc = dict()
    for index in hc_path:
        path_hc[index] = index2desti[index]        
    print(f'minimum of objective function value by {method_list[0]} in {iteration} iteration:', hc_best_dist[-1],'\n',
          'with best path', path_hc)  
     
    #random_walk
    rw_best_dist, rw_path = random_walk_dist(dist_table, iteration)
    np.savez(os.path.join(stored_path,'rw_results.npz'), dist = rw_best_dist, path = rw_path)
    #random_walk results
    path_rw = dict()
    for index in rw_path:
        path_rw[index] = index2desti[index]
    print(f'minimum of objective function value by {method_list[1]} in {iteration} iteration:',rw_best_dist[-1], '\n',
          'with best path', path_rw)
    
    #simu_anneal
    sa_best_dist, sa_path = simu_anneal_dist(dist_table, iteration)
    np.savez(os.path.join(stored_path,'sa_results.npz'), dist = sa_best_dist, path = sa_path)
    #simu_anneal results
    path_sa = dict()
    for index in sa_path:
        path_sa[index] = index2desti[index]
    print(f'minimum of objective function by {method_list[2]} in {iteration} iteration:', sa_best_dist[-1], '\n',
          'with best path', path_sa)
    
    #tabu search
    tb_best_dist, tb_route = tabu_search_dist(dist_table, iteration) 
    np.savez(os.path.join(stored_path,'tb_results.npz'), dist = tb_best_dist, path = tb_route[-1])
    #tabu_search results
    path_tabu = dict()
    for index in tb_route[-1]:
        path_tabu[index] = index2desti[index]
    print(f'minimum of objective function by {method_list[3]} in {iteration} iteration:', tb_best_dist[-1], '\n',
          'with best path', path_tabu)
    #PSO
    PSO_best_dist, PSO_path = PSO_dist(dist_table, PSO_iter)
    np.savez(os.path.join(stored_path,'PSO_results.npz'), dist = PSO_best_dist, path = PSO_path)
    #PSO results
    path_PSO = dict()
    for index in PSO_path:
        path_PSO[index] = index2desti[index]
    print(f'minimum of objective function by {method_list[4]} in {PSO_iter} iteration:', PSO_best_dist[-1], '\n',
          'with best path', path_PSO)     
    
    # Genetic Algorithms
    GA_result = GA(dist_table, GA_iter)  
    np.save(os.path.join(stored_path,'GA_results.npy'), GA_result)
    #GA results        
    path_GA = dict()
    for index in GA_result[-1][0]:
        path_GA[index] = index2desti[index]
    print(f'minimum of objective function by {method_list[5]} in {GA_iter} iteration:', GA_result[-1][1],'\n',
          'with best path', path_GA)
    
    #ant-colony 
    ant_result = run_ant_colony(dist_table, ant_iter)
    np.save(os.path.join(stored_path,'ant_results.npy'), ant_result)
    #ant results
    path_ant = dict()
    for index in ant_result[-1][0]:
        path_ant[index] = index2desti[index]
    print(f'minimum of objective function by {method_list[6]} in {ant_iter} iteration:', ant_result[-1][1],'\n',
          'with best path', path_ant)
    
    #hybrid algorithm (PSO-TS)
    hyb_best_dist, hyb_path = Hybrid_PSO_TS(dist_table, PSO_dist, tabu_search_second, first_stg, second_stg)
    np.savez(os.path.join(stored_path,'hyb_results.npz'), dist = hyb_best_dist, path = hyb_path)
    #hybrid results
    path_hyb = dict()
    for index in hyb_path:
        path_hyb[index] = index2desti[index]
    print(f'minimum of objective function by {method_list[7]} in {first_stg+second_stg} iteration:', hyb_best_dist[-1], '\n',
          'with best path', path_hyb)
     
if __name__ == '__main__':
    run_expr(method_name, **params)
