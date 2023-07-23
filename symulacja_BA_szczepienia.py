from copy import copy
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from numba import njit,jit
from tqdm import tqdm
import matplotlib.ticker as mtick
import copy
from scipy.stats import moment
import sys

plt.rcParams["figure.figsize"] = [10, 10]
plt.rcParams['font.size'] = '13'

def symulacja(node_list, g, state, mu, beta, T, g_degrees):
        
    s_list = np.zeros(T)
    i_list = np.zeros(T)
    r_list = np.zeros(T)
    
    s_list[0] = Counter(state)['S']
    i_list[0] = Counter(state)['I']
    r_list[0] = Counter(state)['R']
    
    for i in tqdm(range(1, T)):
        
        if i <= 50:
            do_one_full_step_with_vacc(node_list, g, state, beta, mu, g_degrees)
        else:
            do_one_full_step(node_list, g, state, beta, mu)
            
        s_list[i] = Counter(state)['S']
        i_list[i] = Counter(state)['I']
        r_list[i] = Counter(state)['R']
        
    return s_list, i_list, r_list

def do_one_full_step(node_list, g, state, beta, mu):
    
    i_count = node_list[state == 'I'].shape[0]
    s_count = node_list[state == 'S'].shape[0]
    
    if (i_count > 0) and (s_count != 0):
        
        nodes_order = np.random.permutation(node_list)
        
        for node in nodes_order:
        
            if state[node] == "I":
                
                recovery_prob = np.random.random()
                
                if recovery_prob <= mu:
                    state[node] = 'S'
                
            elif state[node] == "S":
            
                neighbours = np.array(g[node])
            
                if neighbours.shape[0] > 0:
                    
                    i_neighbours = neighbours[state[neighbours] == 'I']
                    
                    i_count = i_neighbours.shape[0]
                    
                    if i_count > 0:
                        
                        infection_prob = np.random.random_sample(size = i_count)
                        
                        if all(i_p > beta for i_p in infection_prob):
                            pass
                        else:
                            state[node] = 'I'

def do_one_full_step_with_vacc(node_list, g, state, beta, mu, g_degrees):
    
    i_count = node_list[state == 'I'].shape[0]
    s_count = node_list[state == 'S'].shape[0]
    
    if (i_count > 0) and (s_count != 0):
        
        nodes_order = np.random.permutation(node_list)
        
        for node in nodes_order:
        
            vacc_prob = np.random.random()

            if state[node] == 'R':
                pass
            elif vacc_prob <= get_vacc_probability1(node, g_degrees):
                state[node] = 'R'
            else:
                if state[node] == "I":
                    
                    recovery_prob = np.random.random()
                    
                    if recovery_prob <= mu:
                        state[node] = 'S'
                    
                elif state[node] == "S":
                
                    neighbours = np.array(g[node])
                
                    if neighbours.shape[0] > 0:
                        
                        i_neighbours = neighbours[state[neighbours] == 'I']
                        
                        i_count = i_neighbours.shape[0]
                        
                        if i_count > 0:
                            
                            infection_prob = np.random.random_sample(size = i_count)
                            
                            if all(i_p > beta for i_p in infection_prob):
                                pass
                            else:
                                state[node] = 'I'
                            
                            
def get_vacc_probability1(node, g_degrees):
    return 0.0035

def get_vacc_probability2(node, g_degrees):
    return (0.35*np.exp(g_degrees[node])/g_degrees.shape[0])
   
N = 10000
m = 2
p = 2*m/N

sym_steps = 10

p0_infect = 0.2

T = 100

mu = 0.25 # prawd. wyzdrowienia
beta = 0.1 # prawd. zarażenia
 
lambd = beta / mu

t = np.arange(T)

s_list_total = np.zeros(T)
i_list_total = np.zeros(T)
r_list_total = np.zeros(T)

states = np.array(['S', 'I', 'R'])

g = nx.barabasi_albert_graph(N, m)

g_degrees = np.array(list(dict(g.degree).values()))

node_list = np.array(g.nodes())

starter_nodes = node_list[g_degrees <= g_degrees.mean()]

first_moment = np.mean(g_degrees)
second_moment = np.mean(g_degrees**2)
#lambda_c = first_moment/second_moment

print(f'2/mlnN: {round(2/(m*np.log(N)),3)}')
print(f'lambda: {round(lambd,3)}')

for step in tqdm(range(sym_steps)):
    
    g1 = copy.deepcopy(g)
    
    state = np.full(N, 'S')
    state[starter_nodes] = np.random.choice(states[:-1], starter_nodes.shape[0], p=[1- p0_infect, p0_infect])
    
    s_list, i_list, r_list = symulacja(node_list, g, state, mu, beta, T, g_degrees)
    
    s_list_total += s_list/N
    i_list_total += i_list/N
    r_list_total += r_list/N
    
s_list_total = s_list_total/sym_steps
i_list_total = i_list_total/sym_steps
r_list_total = r_list_total/sym_steps

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.plot(t, s_list_total, label = 'S', color='blue')
ax.plot(t, i_list_total, label = 'I', color='red')
ax.plot(t, r_list_total, label = 'R', color='green')

ax.set_yticks(np.linspace(0, 1, 11))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

plt.legend()
plt.title(f'Wyniki dla szczepień jednorodnych dla sieci BA: ' + r'$\lambda$ = ' + f'{round(lambd, 3)}')
plt.xlabel('t [krok]')
plt.show()
plt.close()