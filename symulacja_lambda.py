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

def symulacja(node_list, g, state, mu, beta, T):
    
    for i in tqdm(range(1, T)):
        
        do_one_full_step(node_list, g, state, beta, mu)
    
    i_number = Counter(state)['I']
        
    return i_number

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
                
def get_infected_number(node_list, g, mu, beta, T, symulations):
    
    i_number = 0
    
    for sym in tqdm(range(symulations)):
    
        g1 = copy.deepcopy(g)
        
        state = np.full(N, 'S')
        
        state[starter_nodes] = np.random.choice(states, starter_nodes.shape[0], p=[1- p0_infect, p0_infect])

        i_number += symulacja(node_list, g1, state, mu, beta, T)
        
    return i_number/symulations

N = 1000
m = 2
p = 2*m/N

symulations = 10
p0_infect = 0.2
T = 60

beta = 0.1 # prawd. zarażenia

states = np.array(['S', 'I'])

g = nx.erdos_renyi_graph(N, p)

g_degrees = np.array(list(dict(g.degree).values()))

node_list = np.array(g.nodes())
starter_nodes = node_list[g_degrees <= g_degrees.mean()]

mu_list = np.linspace(0.2, 0.95, 20)
i_list = np.zeros(mu_list.shape[0])
lambda_list = np.zeros(mu_list.shape[0])

first_moment = np.mean(g_degrees)
second_moment = np.mean(g_degrees**2)
lambda_c = 1/first_moment

print(f'lambda_c ER: {round(lambda_c,3)}')

for x in tqdm(range(mu_list.shape[0])):
    
    lambda_list[x] = beta / mu_list[x]
    i_list[x] = get_infected_number(node_list, g, mu_list[x], beta, T, symulations)/N

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_yticks(np.linspace(0, 1, 11))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

ax.plot(lambda_list, i_list, label = 'ER', color = 'blue')
plt.axvline(lambda_c, color = 'blue', linestyle='dashed', alpha = 0.5)
plt.text(lambda_c, -0.06,r'$\lambda_{ER}$',color = 'blue')

g = nx.barabasi_albert_graph(N, m)

g_degrees = np.array(list(dict(g.degree).values()))

node_list = np.array(g.nodes())
starter_nodes = node_list[g_degrees <= g_degrees.mean()]

i_list = np.zeros(mu_list.shape[0])

first_moment = np.mean(g_degrees)
second_moment = np.mean(g_degrees**2)
lambda_c = 2/(m*np.log(N))

print(f'lambda_c BA: {round(lambda_c,3)}')

for x in tqdm(range(mu_list.shape[0])):
    
    i_list[x] = get_infected_number(node_list, g, mu_list[x], beta, T, symulations)/N
    
ax.plot(lambda_list, i_list, label = 'BA', color = 'green')
plt.axvline(lambda_c, color = 'green', linestyle='dashed',  alpha = 0.5)
plt.text(lambda_c, -0.06,r'$\lambda_{BA}$',color = 'green')

plt.title(f'Zależność stacjonarnej ilości chorych osób od tempa epidemii')
plt.xlabel(r'$\lambda$')
plt.ylabel('Względna ilosć osób chorych')
plt.legend()
plt.show()
plt.close()