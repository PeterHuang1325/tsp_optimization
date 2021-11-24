# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 08:33:44 2021

@author: user
"""

params = {
    'iteration': 1000, #iteration for single state methods
    'PSO_iter' : 50, 
    'GA_iter': 100,
    'ant_iter': 50,
    'first_stg' : 40, #for hybrid method
    'second_stg' : 200 
    }

method_name = ['hill_climbing', 'random_walk', 'simu_anneal', 'tabu_search', 'PSO',
               'GA', 'ant_colony', 'hybrid(PSO-TS)']  