import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
import operator
import random
from collections import defaultdict

def find_communities(G,T,r):

    ##Stage 1: Initialization
    memory = {i:{i:1} for i in G.nodes()}
    
    ##Stage 2: Evolution
    for t in range(T):

        listenersOrder = list(G.nodes())
        np.random.shuffle(listenersOrder)

        for listener in listenersOrder:
            speakers = G[listener].keys()
            if len(speakers)==0:
                continue

            labels = defaultdict(int)

            for j, speaker in enumerate(speakers):
                # Speaker Rule
                total = float(sum(memory[speaker].values()))
                labels[list(memory[speaker].keys())[np.random.multinomial(1,[freq/total for freq in memory[speaker].values()]).argmax()]] += 1

            # Listener Rule
            acceptedLabel = max(labels, key=labels.get)

            # Update listener memory
            if acceptedLabel in memory[listener]:
                memory[listener][acceptedLabel] += 1
            else:
                memory[listener][acceptedLabel] = 1


    ## Stage 3:
    for node, mem in memory.iteritems():
        for label, freq in mem.items():
            if freq/float(T+1) < r:
                del mem[label]


    # Find nodes membership
    communities = {}
    for node, mem in memory.iteritems():
        for label in mem.keys():
            if label in communities:
                communities[label].add(node)
            else:
                communities[label] = set([node])


    # Remove nested communities
    nestedCommunities = set()
    keys = communities.keys()
    for i, label0 in enumerate(keys[:-1]):
        comm0 = communities[label0]
        for label1 in keys[i+1:]:
            comm1 = communities[label1]
            if comm0.issubset(comm1):
                nestedCommunities.add(label0)
            elif comm0.issuperset(comm1):
                nestedCommunities.add(label1)
    
    for comm in nestedCommunities:
        del communities[comm]

    return communities

#Read edgelist and make nodes list
file = open("network.dat", "r")
nodes = []
for line in file:
    nodes.append(int(line.split()[0]))
    nodes.append(int(line.split()[1]))
nodes = list(set(sorted(nodes)))

#Function to remove duplicate entries
def Remove(duplicate):
    final_list = []
    for num in duplicate:
        if num not in final_list:
            final_list.append(num)
    return final_list

nodes = Remove(nodes)
sz = nodes[len(nodes) - 1]
edge_list = []
for i in range(sz):
    a = []
    edge_list.append(a)
file = open("network.dat", "r")
for line in file:
    x = int(line.split()[0])
    y = int(line.split()[1])
    edge_list[x-1].append(y-1)
#print(edge_list)
G = nx.Graph()
for i in range(len(edge_list)):
    for j in edge_list[i]:
        if G.has_edge(i,j):
            continue
        else:
            G.add_edge(i,j)
nx.draw_networkx(G,with_labels=True)
plt.show()
nodes = list(G.nodes())
flag = 2
print("Enter node to hide:")
node_to_hide = int(input())
print("Enter budget:")
b = int(input())
ll_b =[]
ll_c =[]
ll_d =[]

#Function to compute ranks of nodes
def rankify(A):

    # Rank Vector
    R = [0 for x in range(len(A))]
    sorted_x = sorted(A.items(), key=operator.itemgetter(1))
    R[sorted_x[0][0]] = 1
    for i in range(len(sorted_x) - 1):
        for j in range(i+1,len(sorted_x)):
            if sorted_x[i][1] == sorted_x[j][1]:
                R[sorted_x[j][0]] = R[sorted_x[i][0]]
            else:
                R[sorted_x[j][0]] = R[sorted_x[i][0]] + 1
    # Return Rank Vector
    return R

#ROAM Heuristic for required number of iterations (Node (Individual) hiding)
while flag>=0:
    nh = list(G.neighbors(node_to_hide))#Neighbours of node to hide(v+)
    max_deg_elem = nh[0]

    #Computing centrality measures
    d_c_dict = nx.degree_centrality(G)
    rr = rankify(d_c_dict)
    ll_d.append(rr[node_to_hide])

    c_c_dict = nx.closeness_centrality(G)
    rr = rankify(c_c_dict)
    ll_c.append(rr[node_to_hide])

    b_c_dict = nx.betweenness_centrality(G)
    rr = rankify(b_c_dict)
    ll_b.append(rr[node_to_hide])

    for a in nh:
        if G.degree(a) > G.degree(max_deg_elem):
            max_deg_elem = a
    G.remove_edge(node_to_hide,max_deg_elem) #Remove edge between v+ and max degree element (v0)
    nb_node_hide = list(G.neighbors(node_to_hide)) #Neighbours of v+
    nb_max_elem = list(G.neighbors(max_deg_elem)) #Neighbours of v0
    nb_node_hide = list(set(nb_node_hide) - set(nb_max_elem)) #Neighbours of v+ but not v0
    if len(nb_node_hide)<b:
        for a in nb_node_hide:
                G.add_edge(max_deg_elem,a)
    else:
        while len(nb_node_hide) > b-1:
            maxdeg = 0
            for i in range(len(nb_node_hide)):
                if(G.degree(nb_node_hide[i]) > G.degree(nb_node_hide[maxdeg]) ):
                    maxdeg = i
            nb_node_hide.remove(nb_node_hide[maxdeg])
        for a in nb_node_hide:
            G.add_edge(max_deg_elem,a)
    #nx.draw_networkx(G,with_labels=True)
    #plt.show()
    flag-=1

# create some x data and some integers for the y axis
y1 = np.array(ll_d)
x1 = np.arange(3)
plt.subplot(3, 1, 1)
plt.plot(x1, y1, 'o-')
plt.title('Degree Centrality')
plt.ylabel('rankings')
plt.xlabel('no. of iterations')

y2 = np.array(ll_c)
x2 = np.arange(3)

plt.subplot(3, 1, 2)
plt.plot(x2, y2, '.-')

plt.title('Closeness Centrality')
plt.ylabel('rankings')
plt.xlabel('no. of iterations')

y3 = np.array(ll_b)
x3 = np.arange(3)

plt.subplot(3, 1, 3)
plt.plot(x3, y3, '*-')

plt.title('Betweenness Centrality')
plt.ylabel('rankings')
plt.xlabel('no. of iterations')
plt.tight_layout()
plt.show()



#Community Hiding

cs1 = find_communities(G,25,0.25)
cs = []
for key, value in cs1.iteritems():
    cs.append(value)

group = [int(x) for x in input().split()]

#disconnect d<=b links iinside the community
b = int(input())
d = int(input())
flag = 3
alpha = 0.5

def intersecting_communities(x,y):
    num = 0.0
    for a in x:
        if len(list(set(a).intersection(y))):
            num += 1.0
    return num

def max_intersecting_community_size(x,y):
    num = 0.0
    for a in x:
        intersection_len = len(list(a).intersection(y))
        if intersection_len > num:
            num = float(intersection_len)
    return num
def number_of_common_node(x,y):
    num = 0.0
    for a in x:
        for b in y:
            if a == b:
                num += 1.0
    return num

#find length of group
#print("hj")
ll_mu = []
while flag:
    while d:
        v1,v2 = random.sample(range(0,len(group)-1),2)
        print(v1,v2)
        if(G.has_edge(group[v1],group[v2])):
            G.remove_edge(group[v1],group[v2])
            d-=1
    rem = b-d
    n = G.number_of_nodes()
    print(n)
    while rem:
        v1 = random.sample(range(0,len(group)-1),1)
        v2 = random.sample(range(0,n-1),1)
        print(v1,v2)
        if G.has_edge(group[int(v1[0])],int(v2[0])):
            continue
        elif v2 in group:
            continue
        else:
            G.add_edge(group[v1[0]],v2[0])
            rem-=1
    flag -= 1
    mu_ = (intersecting_communities(cs,group)-1) / ((max(len(cs)-1,1)) * max_intersecting_community_size(cs,group))
    _mu_ = 0.0
    max_fact = max((len(nodes) - len(group)),1)
    for c in cs:
        _mu_ +=  ( float(number_of_common_node(c,group)) / float(max_fact) )

    mu = ( alpha * (mu_) ) + ( (1-alpha) * (_mu_) )
    ll_mu.append(mu)

y1 = np.array(ll_d)
x1 = np.arange(4)
plt.subplot(1, 1, 1)
plt.plot(x1, y1, 'o-')
plt.title('Concealment Factor')
plt.ylabel('mu')
plt.xlabel('no. of iterations')