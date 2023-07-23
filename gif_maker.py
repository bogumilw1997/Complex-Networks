import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from numba import njit,jit
from tqdm import tqdm
import matplotlib.ticker as mtick
import imageio
import os
import sys
import copy

plt.rcParams["figure.figsize"] = [10, 10]
plt.rcParams['font.size'] = '13'

def symulacja(node_list, g, state, mu, beta, T):
        
    s_list = np.zeros(T)
    i_list = np.zeros(T)
    
    s_list[0] = Counter(state)['S']
    i_list[0] = Counter(state)['I']
    
    for i in tqdm(range(1, T)):
        
        do_one_full_step(node_list, g, state, beta, mu)
        
        s_list[i] = Counter(state)['S']
        i_list[i] = Counter(state)['I']
        
    return s_list, i_list

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
        

N = 200
m = 2
p = 2*m/N

p0_infect = 0.2

T = 60

mu = 0.25 # prawd. wyzdrowienia
beta = 0.1 # prawd. zarażenia

lambd = beta / mu

t = np.arange(T)

states = np.array(['S', 'I'])
color_state_map = {'S': 'blue', 'I': 'red'}

g = nx.erdos_renyi_graph(N, p)
#g = nx.barabasi_albert_graph(N, m)

g_degrees = np.array(list(dict(g.degree).values()))

node_list = np.array(g.nodes())

state = np.full(N, 'S')

starter_nodes = node_list[g_degrees <= g_degrees.mean()]
state[starter_nodes] = np.random.choice(states, starter_nodes.shape[0], p=[1- p0_infect, p0_infect])

state_dict = dict(zip(node_list, state))

pos = nx.spring_layout(g, k=9/np.sqrt(N))

fname = f'{0}.png'
dirname = 'semestr2/MSZ/projekt/wersja2/shots/'

nx.draw(g, pos=pos, with_labels=True, font_weight='bold', font_color='white', node_color=[color_state_map[s] 
                    for s in state], nodelist=node_list, node_size=[d*100  for d in g_degrees], alpha=0.8, edge_color='grey', labels = state_dict)

plt.savefig(dirname + fname)
plt.close()

filenames = []

filenames.append(fname)

s_list = np.zeros(T)
i_list = np.zeros(T)

s_list[0] = Counter(state)['S']
i_list[0] = Counter(state)['I']

for i in tqdm(range(1, T)):
    
    do_one_full_step(node_list, g, state, beta, mu)
    
    s_list[i] = Counter(state)['S']
    i_list[i] = Counter(state)['I']
    
    state_dict = dict(zip(node_list, state))
    
    fname = f'{i}.png'
    
    nx.draw(g, pos=pos, with_labels=True, font_weight='bold', font_color='white', node_color=[color_state_map[s] 
                    for s in state], nodelist=node_list, node_size=[d * 100 for d in g_degrees], alpha=0.8, edge_color='grey', labels = state_dict)
    
    plt.savefig(dirname + fname)
    plt.close()
    
    filenames.append(fname)

print('Preparing .gif file ...')

with imageio.get_writer('semestr2/MSZ/projekt/wersja2/sis.gif', mode='I', fps = 2) as writer:
    for filename in filenames:
        image = imageio.imread(dirname + filename)
        writer.append_data(image)

print('Deleting .png files ...')

for filename in set(filenames):
    os.remove(dirname + filename)

print('Preparing graph ...')

s_list_total = s_list/N
i_list_total = i_list/N
   
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.plot(t, s_list_total, label = 'S', color='blue')
ax.plot(t, i_list_total, label = 'I', color='red')

ax.set_yticks(np.linspace(0, 1, 11))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

plt.legend()
plt.title('Względna ilość osób w każdej grupie modelu SIS')
plt.xlabel('t [krok]')
#plt.show()

plt.savefig('semestr2/MSZ/projekt/wersja2/sis.png')
plt.close()