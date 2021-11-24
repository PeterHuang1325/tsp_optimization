# -*- coding: utf-8 -*-
"""
@author: Pin-Han, Huang
"""
import numpy as np
import itertools

# In[1]:
# define some functions

def init_visit(table):
    init_num = np.random.randint(0,len(table),1) #select initial index
    
    #set index range to be permutated
    idx_range = np.concatenate((np.arange(0, init_num), np.arange(init_num + 1,36)))
    rand_vis = np.random.choice(idx_range, len(table) - 1, replace = False)  
        
    return np.pad(rand_vis, (1,1), 'constant', constant_values= init_num)
   
def calc_dist(table, path):
    tot_dist = 0
    for i in range(len(path) - 1):
        tot_dist += table[path[i]][path[i+1]]
    return tot_dist
        
#define swap function for constructing domain
def swap_loc(path):
    swapped = [] #store all swapped neighbors
    for i in range(1, len(path) - 1): #1~14
        for j in range(i + 1, len(path) - 1): #1~14
            neighbor = path.copy()
            neighbor[i] = path[j]
            neighbor[j] = path[i]
            swapped.append(neighbor)
    return swapped


# In[2]:
#hill climbing
def hill_climbing_dist(table, iteration):
    hc_best_dist = [] #hc_best list
    np.random.seed(100)
    path = init_visit(table)
    min_dist = calc_dist(table, path)
    
    i = 0
    seed = 100
    while  i < iteration:
        np.random.seed(i + seed)
        domain = swap_loc(path)
        rand_swap = np.random.choice(len(domain)) #choose the index of swap domain
        new_path = domain[rand_swap] #get the index from numpy array and extract path from domain
        new_dist = calc_dist(table, new_path)
        
        if new_dist < min_dist:
            path = new_path
            min_dist = new_dist
            
        hc_best_dist.append(min_dist) #store y_b across iteration
        i += 1     
    return hc_best_dist, path


    

# In[3]:
# random_walk
def random_walk_dist(table, iteration):
    rw_best_dist = [] #rw_best list
    np.random.seed(100)
    path = init_visit(table)
   
    min_dist = calc_dist(table, path)
    
    i = 0
    seed = 100
    while  i < iteration:
        np.random.seed(i + seed)
        domain = swap_loc(path)
        rand_swap = np.random.choice(len(domain)) #choose the index of swap domain
        path = domain[rand_swap] #get the index from numpy array and extract path from domain
        new_dist = calc_dist(table, path)
        
        if new_dist < min_dist:
            min_dist = new_dist
            
        rw_best_dist.append(min_dist) #store y_b across iteration        
        i += 1     
    return rw_best_dist, path



# In[4]:
#simulated annealing setup
def Boltzmann(df, T):
    epsilon =  1e-10
    return np.exp(-df / (T + epsilon))

def get_temp(t, T):
    epsilon = 1e-10
    return np.where(t > 120, np.divide(T, np.log(t + epsilon)), T)

def init_temp(table, seed):
    
    temp_list = []
    for i in range(4):
        np.random.seed(seed + 10*i)
        init_path = init_visit(table)
        temp_list.append(calc_dist(table, init_path))
    return np.mean(temp_list)

# In[5]:
#simulated annealing    
def simu_anneal_dist(table, iteration):
    sa_best_dist = [] #sa_best list
    seed = 100
    np.random.seed(100)
    #define init path
    path = init_visit(table) #numpy array
    min_dist = cur_dist = calc_dist(table, path)
    #initial temperature
    T = init_temp(table, seed)
    
    i = 0
    while i < iteration:
        np.random.seed(seed + i)
        domain = swap_loc(path)
        rand_swap = np.random.choice(len(domain))  
        new_path = domain[rand_swap] #get the index from numpy array and extract path from domain
        new_dist = calc_dist(table, new_path)
        
        df_dist = new_dist - min_dist
        prob = np.minimum(1, Boltzmann(df_dist, get_temp(i,T))) #detect if df_y > 0 -> metropolis criterion
        if prob < 1:
            if prob > np.random.rand():
                cur_dist = new_dist
                path = new_path
        else:
            cur_dist = new_dist
            path = new_path
            if cur_dist < min_dist:
                min_dist = cur_dist
        sa_best_dist.append(min_dist)
        i += 1
    return sa_best_dist, path



# In[6]:
#tabu search

def tabu_search_dist(table, iteration):
    np.random.seed(100)
    #store best distance
    tb_best_dist = []
    #store best route
    tb_route = []
    #create tabu_list and set tenure
    tabu_list = []
    tabu_tenure = 10
    
    
    #create initial
    path = init_visit(table) #numpy array
    best_path = cur_path = path
    min_dist = calc_dist(table, path)
    
    i = 0
    seed = 100
    while i < iteration:
        np.random.seed(i + seed)
        domain = swap_loc(cur_path) #swap cur_path in each iteration
        new_path = domain[0]
        move = [] #initial move
        count_cond = 0
        
        for n in range(len(domain)):
            
            new_dist = calc_dist(table, new_path)
            test_dist = calc_dist(table, domain[n])
            
            cond = (move not in tabu_list) & ((test_dist < new_dist) | (test_dist < min_dist))
            if cond == True:
                new_path = domain[n]
                move = np.where(new_path != cur_path)[0].tolist() #tuple with a numpy array in it and converge to list    
                count_cond += 1 
        
        if count_cond > 1:   # exclude the condition of empty list move; if count_cond = 1, no more better search   
            cur_path = new_path
            cur_dist = calc_dist(table, cur_path)
        
            if cur_dist < min_dist:
                best_path = cur_path
                min_dist = calc_dist(table, best_path)
                
            #append move to tabu_list
            tabu_list.append(move)
            
            if len(tabu_list) >= tabu_tenure:
                tabu_list.remove(tabu_list[0]) #FIFO
        
        
        asp_dict = dict()
        #Aspiration criterion: expectation improvement
        for tb in range(len(tabu_list)):
            tab_path = cur_path.copy()
            tab_copy = tab_path[tabu_list[tb][0]].copy()              
            tab_path[tabu_list[tb][0]] = tab_path[tabu_list[tb][1]] #[[2,3], [1,4]] #list indexing
            tab_path[tabu_list[tb][1]] = tab_copy
            tab_dist = calc_dist(table, tab_path)
            if tab_dist < min_dist:
                asp_dict[tab_dist] = tab_path
                            
               
        if len(asp_dict) > 0:        
            min_dist = np.min([k for k in asp_dict.keys()])
            cur_path = asp_dict[min_dist] #set new cur_path
            best_path = cur_path #set optima best_path = cur_path
            
        
        tb_best_dist.append(min_dist)
        i += 1
    #store best_path    
    tb_route.append(best_path)
    return tb_best_dist, tb_route



# In[7]:
#PSO setup
def init_pop(seed, table, num_partkl):
    ptkl_list = [] #length: 20, [np.array(), np.array(),....]
    for p in range(num_partkl):
        np.random.seed(seed + p)
        ptkl = init_visit(table)
        ptkl_list.append(ptkl)
    return ptkl_list

def pop_swap(path_list):
    swap_pop = []
    for path in path_list:
         swap_pop.append(swap_loc(path))
    return swap_pop

def pop_calc(table, path_list):
    calc_pop = []
    for path in path_list:
         calc_pop.append(calc_dist(table, path))
    return calc_pop

#set the range of velocity
def set_v(v, v_range):
    if v < v_range[0]:
        v = v_range[0]
    elif v > v_range[1]:
        v = v_range[1]
    else:
        v = v     
    return v 

# In[8]:
#PSO
def PSO_dist(table, iteration):
    seed = 100
    num_particles = 20 #number of particles
    
    path_list = init_pop(seed, table, num_particles)           
    t = 0
    v = [30]*len(path_list) #[20,20, ...,20], length = 20
    v_range = np.array([10,50])
    
    pos_loc = path_list #shape(20,) #pos_loc: local best position; x: current position
    #v_range = np.delete(np.arange(-3,4), 3) #-3, -2, -1, 1, 2, 3 (delete the third idx)
    pos_glob = path_list[0]
    PSO_best = []
    while t < iteration:
        domain = pop_swap(path_list)
        f_1 = pop_calc(table, path_list) #shape(20,)
        #pos_glob = path_list[np.where(f_1 == np.min(f_1))[0][0]] #tuple->array->index
        
        np.random.seed(seed + 100*t)
        #rand = np.random.choice(v_range, num_particles) #shape(20,)
        
        new_path_list = []
        for idx, path in enumerate(path_list):
            
            #print(domain[idx])
            np.random.seed(seed + 100*idx)
            rand = np.random.choice(np.arange(0, len(domain[idx])-1), v[idx] , replace = False) #(0, 595)
            
            #generate candidate dictionary w.r.t the velocity
            candidate = dict()
            for rd in rand:            
                neighbor = domain[idx][rd]
                val = calc_dist(table, neighbor)
                if val < calc_dist(table, path):
                    candidate[val] = neighbor
      
            if len(candidate) > 0:
                target =  np.min([k for k in candidate.keys()])
                new_path_list.append(candidate[target]) #get the minimum neighbor
                #set the velo to original + length of candidate
                v[idx] = v[idx] + len(candidate)
                # meet the range of velo
                v[idx] = set_v(v[idx], v_range)
            else:
                #set the velo as original + iteration
                v[idx] = v[idx] + t 
                v[idx] = set_v(v[idx], v_range)
                new_path_list.append(pos_glob) 
        f_2 = pop_calc(table, new_path_list) #shape(20,)
        
        #only stores the best
        if (len(PSO_best) == 0) or (np.min(f_2) < PSO_best[-1]):
            PSO_best.append(np.min(f_2))
            pos_glob = new_path_list[np.where(f_2 == np.min(f_2))[0][0]]
        else:
            PSO_best.append(PSO_best[-1])
        
        if len(np.where(f_2 < f_1)[0]) > 0: #exclude the f_1.all() > f_2.all() case
            pos_loc[np.where(f_2 < f_1)[0][0]] = new_path_list[np.where(f_2 < f_1)[0][0]]
        path_list = new_path_list
        t += 1
        
    return PSO_best, pos_glob


# In[9]:
# GA setup
# roulette-wheel selection
def selection(table, pool):
    mate_size = 6
    seed = 100
    #pool, req_set = population(data,w) #list: shape(10, 15)
    seq_point = []
    
    for seq in pool:
        #[[seq, total_points], [],[]]
        seq_point.append([seq] + [np.reciprocal(calc_dist(table, seq))])
        
    sort_seq_point = sorted(seq_point, key = lambda seq_point: seq_point[1])

    cum_list = []
    cum_sum = 0
    func_sum = np.sum([v[1] for v in sort_seq_point])
    
    for points in sort_seq_point:
        prob = (points[1] / func_sum)
        cum_sum += prob
        cum_list.append([points[0]] + [cum_sum])    
        
    mate_pool = []
    rnd = 0   
    while len(mate_pool) < mate_size:
        for pb in cum_list:
            np.random.seed(seed+ 100*rnd)
            rand_num = np.random.rand()
            if (rand_num <  pb[1]) and (len(mate_pool) < mate_size):
                #print(pb[0].copy())
                mate_pool.append(pb[0].copy())                     
        rnd += 1
        
    return mate_pool


def crossover(mate):
    mate_idx = [k for k in range(len(mate))]  #[0,1,2,3]  
    comb = list(itertools.combinations(mate_idx, 2)) #[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]                
    seed = 100
    cross_pool = []
    for c in range(len(comb)):
        np.random.seed(seed + 100*c)
        #random the crossover position
        cross_pos_a = np.random.randint(1,len(mate[0])-1) #random: 1~35
        #get the store index to be crossed
        cross_loc = mate[comb[c][0]][cross_pos_a]
        #delete the store in a 
        mate_del_a = np.delete(mate[comb[c][0]], cross_pos_a)
        #print(mate[comb[c][0]].shape)
        #find the corresponding position of store index in b
        
        cross_pos_b = np.where(mate[comb[c][1]] == cross_loc)[0][0]
        #print(cross_pos_b)
        #delete the store in b
        mate_del_b = np.delete(mate[comb[c][1]], cross_pos_b)
        
        #crossover to insert the position of store index in a and b 
        cross_a = np.insert(mate_del_a, cross_pos_b, cross_loc)
        cross_b = np.insert(mate_del_b, cross_pos_a, cross_loc)

        cross_pool.append(cross_a.copy())
        cross_pool.append(cross_b.copy())
    return cross_pool
    
# mutate: mutate swap
def mutate(cross):
    seed = 100
    mutate_pool = []
    for idx, m in enumerate(cross):
       domain = swap_loc(m)
       np.random.seed(seed + 100*idx)
       mutate_idx = np.random.randint(0, len(domain))
       #mutate_pool.append(m.copy())
       mutate_pool.append(domain[mutate_idx].copy())
       
    return mutate_pool #shape(12,15)
 
# In[10]:
# Genetic Algorithm
def GA(table, iteration):

    size = 10
    seed = 100
    pool = init_pop(seed, table, size) #list: shape(10, 37)
    #Use first string of pool as initial best
    best_points = calc_dist(table, pool[0])
    best_route = pool[0]
   
    t = 0
    min_list = []
    while t < iteration:
        mate_pool = selection(table, pool)

        cross_pool= crossover(mate_pool)
        mutate_pool = mutate(cross_pool)
        
        qualify_pool = []
        qualified = []     #[[np.array(),points], [],[]]

        np.random.seed(seed + 100*t)
        q_idx = np.random.choice(np.arange(0,len(mutate_pool)), size, replace= False)
        
        for q in q_idx:
            qualify_pool.append(mutate_pool[q])
            qualified.append([mutate_pool[q]]+[calc_dist(table, mutate_pool[q])])
            
                
        #print(qualified)
        # append max points in each iteration    
        min_points = qualified[0][1]
        best_path = qualified[0][0]
        for q_point in qualified:
            if q_point[1] < min_points:
                min_points = q_point[1]
                best_path = q_point[0]
                    
            # global best for plotting       
        if min_points < best_points:
            best_points = min_points
            best_route = best_path
                
        min_list.append([best_route]+[best_points])
        pool = qualify_pool   

        t += 1
    return min_list



# In[11]:
#ant-colony
class AntColony:
    def __init__(self, pop_size, evaporate_rate, pheromone_factor, table, calc_dist):
        
        self.num_ants = pop_size
        self.table = table
        self.stores = np.arange(0,len(table)).tolist() 
        self.num_stores = len(table)
        self.calc_dist = calc_dist
        self.evaporate_rate = evaporate_rate
        self.pheromone_factor = pheromone_factor
        

    def pheromone_init(self):
        return -np.eye(self.num_stores) + 1 #construct matrix with ones except the diagonal to be zero
    
    def roulette_wheel(self, phero_list):
        
        #phero_sorted = sorted(phero_list)
        
        phero_density = [phero / np.sum(phero_list) for phero in phero_list]
        phero_cum_prob = np.cumsum(phero_density)
        #print(phero_cum_prob)
        rand = np.random.rand()
        for loc, prob in enumerate(phero_cum_prob):
            if rand < prob:
               return loc 
        
    def one_ant(self, seed, phero_map):
        np.random.seed(100)
        #random choose store as first location
        init_num = np.random.randint(0,self.num_stores) 
        phero_copy = phero_map.copy()
        
        location = init_num #initial location
        #set the stored path list
        self.path = [init_num]
        for t in range(self.num_stores-1): #remaining 35 locations: e.g. 0~34
            phero_list = phero_copy[location]
            phero_copy[:, location] = 0 #set location column to zero s.t. node to be non-repeat
            np.random.seed(seed + t)
            node = self.roulette_wheel(phero_list)
            self.path.append(node) #append to path list
            location = node #update the location
        #return to the origin    
        self.path.append(init_num) 

    
    def multi_ant(self, phero_map):
        seed = 100
        self.colony = dict() #[[path, dist],[,],[,],...]
        
        for ant in range(self.num_ants): #0,1,2,3,....num of ants
            self.one_ant(seed + 100*ant, phero_map)
            ant_dist =  self.calc_dist(self.table, self.path)
            self.colony[ant_dist] = self.path
            
        self.min_dist = np.min([k for k in self.colony.keys()])
        self.best_path = self.colony[self.min_dist]
        self.max_dist = np.max([c for c in self.colony.keys()])
        return self.min_dist, self.best_path
    
    def update_pheromone(self, phero_map):       
        phero_map = phero_map * (1 - self.evaporate_rate)  #pheromone_map after evaporation
        num_same_path = self.num_ants - len(self.colony) + 1
        delta_phero =  (self.pheromone_factor * num_same_path) * (self.min_dist / self.max_dist)
        for p in range(len(self.best_path) - 1):
            phero_map[self.best_path[p]][self.best_path[p+1]] += delta_phero #update the vital pheromone
        return phero_map
    
# In[12]:
# run ant_colony
def run_ant_colony(table, iteration): 
    pop_size = 20
    evaporate_rate = 0.1
    pheromone_factor = 2
    solver = AntColony(pop_size, evaporate_rate, pheromone_factor, table, calc_dist)    

    glob_best = []
    phero_map = solver.pheromone_init()    
    for i in range(iteration):
        #print(phero_map)
        min_dist, best_path = solver.multi_ant(phero_map)
        phero_map = solver.update_pheromone(phero_map)
        glob_best.append([best_path] + [min_dist])
    return glob_best

# In[13]:
#prepare for the hybrid algorithm
def tabu_search_second(table, init_path, iteration):
    np.random.seed(100)
    #store best distance
    tb_best_dist = []
    #store best route
    tb_route = []
    #create tabu_list and set tenure
    tabu_list = []
    tabu_tenure = 10
    
    
    #create initial
    path = init_path #numpy array
    best_path = cur_path = path
    min_dist = calc_dist(table, path)
    
    i = 0
    seed = 100
    while i < iteration:
        np.random.seed(i + seed)
        domain = swap_loc(cur_path) #swap cur_path in each iteration
        new_path = domain[0]
        move = [] #initial move
        count_cond = 0
        
        for n in range(len(domain)):
            
            new_dist = calc_dist(table, new_path)
            test_dist = calc_dist(table, domain[n])
            
            cond = (move not in tabu_list) & ((test_dist < new_dist) | (test_dist < min_dist))
            if cond == True:
                new_path = domain[n]
                move = np.where(new_path != cur_path)[0].tolist() #tuple with a numpy array in it and converge to list    
                count_cond += 1 
        
        if count_cond > 1:   # exclude the condition of empty list move; if count_cond = 1, no more better search   
            cur_path = new_path
            cur_dist = calc_dist(table, cur_path)
        
            if cur_dist < min_dist:
                best_path = cur_path
                min_dist = calc_dist(table, best_path)
                
            #append move to tabu_list
            tabu_list.append(move)
            
            if len(tabu_list) >= tabu_tenure:
                tabu_list.remove(tabu_list[0]) #FIFO
        
        
        asp_dict = dict()
        #Aspiration criterion: expectation improvement
        for tb in range(len(tabu_list)):
            tab_path = cur_path.copy()
            tab_copy = tab_path[tabu_list[tb][0]].copy()              
            tab_path[tabu_list[tb][0]] = tab_path[tabu_list[tb][1]] #[[2,3], [1,4]] #list indexing
            tab_path[tabu_list[tb][1]] = tab_copy
            tab_dist = calc_dist(table, tab_path)
            if tab_dist < min_dist:
                asp_dict[tab_dist] = tab_path
                            
               
        if len(asp_dict) > 0:        
            min_dist = np.min([k for k in asp_dict.keys()])
            cur_path = asp_dict[min_dist] #set new cur_path
            best_path = cur_path #set optima best_path = cur_path
            
        
        tb_best_dist.append(min_dist)
        i += 1
    #store best_path    
    tb_route.append(best_path)
    return tb_best_dist, tb_route


# In[14]:
#hybrid algorithm (PSO-TS)
def Hybrid_PSO_TS(table, PSO, TS, first, second):
    
    PSO_first = PSO(table, first) #particles: 20
    hybrid = TS(table, PSO_first[1], second) #pos_glob of PSO
    hybrid_best = PSO_first[0] + hybrid[0] #concat dist results []+[] = [...]
    
    return hybrid_best, hybrid[1][-1] #dist and last path
