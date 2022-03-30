import numpy as np
import pylab as plt
import networkx as nx

# mapeamento de transições entre estados

# primeiro grafo
# transitions_list = [(2,3),(3,2),(3,1),(3,4),(1,3),(1,5),(4,3),(4,5),(5,1),(5,4),(5,5),(0,4),(4,0)]

# segundo grafo
transitions_list = [(0,1),(1,0),(0,4),(4,0),(4,8),(8,4),(8,12),(12,8),(8,9),(9,8),(9,13),(13,9),(13,14),(14,13),(14,10),(10,9),(9,10),(9,5),(5,9),(5,6),(6,5),(6,2),(2,6),(2,3),(3,2),(3,7),(7,3),(7,11),(11,7),(11,15),(15,11)]

# objetivo do primeiro grafo
# goal = 5

# objetivo do segundo grafo
goal = 15

# gerando o grafo para a lista de pontos
G=nx.Graph()
G.add_edges_from(transitions_list)
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G,pos)
nx.draw_networkx_edges(G,pos)
nx.draw_networkx_labels(G,pos)
plt.show()

# taxa de aprendizagem (gamma)
gamma = 0.9

# inicializando matriz de recompensas
# MATRIX_SIZE = 6 # tamanho da matriz referente ao primeiro grafo
MATRIX_SIZE = 16 # tamanho da matriz referente ao segundo grafo
R = np.matrix(np.ones(shape=(MATRIX_SIZE, MATRIX_SIZE)))
R *= -1

for node in transitions_list:
    if node[1] == goal:
        R[node] = 100
    else:
        R[node] = 0

print("Recompensa inicial:")
print(R)

Q = np.matrix(np.zeros([MATRIX_SIZE,MATRIX_SIZE]))

print('Q inicial:')
print(Q)

# retorna os movimentos possiveis a partir de um nó
def available_actions(state):
    current_state_row = R[state,]
    av_act = np.where(current_state_row >= 0)[1]
    return av_act

# escolhe aleatoriamente o proximo movimento baseado na lista de movimentos possiveis
def sample_next_action(available_actions_range):
    next_action = int(np.random.choice(available_act,1))
    return next_action

# atualiza o estado atual com o proximo movimento, atualizando a matriz Q-learning no processo
# Q(state, action) = R(state, action) + Gamma * Max[Q(next state, all actions)]
def update(current_state, action, gamma):
    max_index = np.where(Q[action,] == np.max(Q[action,]))[1]

    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, size = 1))
    else:
        max_index = int(max_index)
    max_value = Q[action, max_index]

    Q[current_state, action] = R[current_state, action] + gamma * max_value
    #print('max_value', R[current_state, action] + gamma * max_value)

    if (np.max(Q) > 0):
        return(np.sum(Q/np.max(Q)*100))
    else:
        return (0)

# treinamento
episodes = 700 
scores = []
for i in range(episodes):
    current_state = np.random.randint(0, int(Q.shape[0]))
    available_act = available_actions(current_state)
    action = sample_next_action(available_act)
    score = update(current_state,action,gamma)
    scores.append(score)

print("Matriz Q treinada:")
# printa a matriz normalizada (em porcentagem)
print(Q/np.max(Q)*100)

plt.plot(scores)
plt.show()

# teste

# initial_state = 2 # nó inicial do primeiro grafo
initial_state = 9 # nó inicial do segundo grafo
current_state = initial_state
steps = [current_state]

while current_state != goal:

    next_step_index = np.where(Q[current_state,]
        == np.max(Q[current_state,]))[1]

    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index, size = 1))
    else:
        next_step_index = int(next_step_index)

    steps.append(next_step_index)
    current_state = next_step_index

print("Melhor caminho: ")
print(steps)