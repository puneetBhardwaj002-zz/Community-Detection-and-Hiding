import collections
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

def edges_in_community(co,graph):
    count = 0
    for i in range(len(co)-1):
        for j in range(i+1,len(co)):
            if graph[co[i]][co[j]] == 1:
                count += 1
    return count

def edges_with_atleast_one_end_in_community(co,graph):
    count = 0
    for vertex in co:
        for neighbour in range(len(graph[vertex])):
            if graph[vertex][neighbour] == 1:
                count += 1
    return count

def calculate_modularity(c,g):
    sum = 0
    for co in c:
        e = edges_in_community(co,g)
        e = float(e) / float(len(g))
        a = edges_with_atleast_one_end_in_community(co,g) 
        a = float(a) / float(2 * len(g)) 
        sum += (e-(a*a))
    return sum

def merge_communities(graph,communities,old_mod):
    max_mod = old_mod
    v1_max = -1
    v2_max = -1
    for i in range(0,len(communities)-1):
        for j in range(i+1,len(communities)):
            flag = 0 
            for x in communities[i]:
                for y in communities[j]:
                    if graph[x][y] == 1:
                        communities_1 = list(communities)
                        l1 = list(communities_1[i])
                        l2 = list(communities_1[j])
                        communities_1.remove(l1)
                        communities_1.remove(l2)
                        communities_1.append(l1 + l2)
                        new_mod = calculate_modularity(communities_1,graph)
                        if new_mod > max_mod:
                            max_mod = new_mod
                            v1_max = i
                            v2_max = j
                        flag = 1
                        break
                if flag:
                    break
    if v1_max != -1 and (v2_max != -1):
        l1 = list(communities[v1_max])
        l2 = list(communities[v2_max])
        communities.remove(l1)
        communities.remove(l2)
        communities.append(l1 + l2)
    return communities,max_mod

if __name__ == '__main__':
    file = open("network.dat", "r")
    nodes = []
    for line in file: 
        nodes.append(int(line.split()[0]))
        nodes.append(int(line.split()[1]))
    nodes = list(set(sorted(nodes)))
    def Remove(duplicate):
        final_list = []
        for num in duplicate:
            if num not in final_list:
                final_list.append(num)
        return final_list
    nodes = Remove(nodes)
    sz = nodes[len(nodes) - 1]
    adj_list = np.zeros((sz,sz), dtype='int32')
    file = open("network.dat", "r")
    for line in file:
        x = int(line.split()[0])
        y = int(line.split()[1])
        adj_list[x-1][y-1] = 1
        adj_list[y-1][x-1] = 1
    H = nx.from_numpy_matrix(adj_list)
    nx.draw_networkx(H,with_labels=True)
    plt.show()
    nodes = list(H.nodes())
    communities=[]
    for node in nodes:
        communities.append([node])
    graph = nx.adjacency_matrix(H).toarray()
    old_mod = calculate_modularity(communities,graph)
    count = 1
    while 1:
        communities,bc = merge_communities(graph,communities,old_mod)
        if(bc > old_mod):
            old_mod=bc
        else:
            print(old_mod)
            print(communities)
            break
color_list = ['green','blue','red','yellow','cyan','magenta','black']
color_map = [None] * len(nodes)
for i in range(len(communities)):
    for node in communities[i]:
        color_map[node] = color_list[i]  
nx.draw(H,node_color = color_map,with_labels = True)
plt.show()