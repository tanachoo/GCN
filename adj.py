
# author: yoshi
# Write start: 5/16/2019

# Make adjacency matrix script


import numpy as np
import time

start_time=time.time()

# store node names as list
# [gene1, gene2,,,,]
print('\nprep node names list...')
node_names=[]
nodedata='./processed_data/target0_20190425/dataset_node.csv'
print('load: {0}'.format(nodedata))
with open(nodedata,'r') as f:
	for line in f:
		nodes=line.strip()
		node_names.append(nodes)

print('#node_names: ',len(node_names))

# store edge pair names as list
# [(gene1,gene2),,,]
print('\nedge pair list...')
pairs=[]
pairdata='./PathwayCommons11.All.hgnc.sif'
print('load: {0}'.format(pairdata))
with open(pairdata,'r') as f:
	for line in f:
		comp=line.strip().split('\t')
		pair=(comp[0],comp[2])
		pairs.append(pair)

print('#pairs: ',len(pairs))
pairs_uniq=list(set(pairs))
print('remove uniq pairs...')
print('#pairs uniqed: ',len(pairs_uniq))

# convert pairs(gene1,gene2) to node id, then store them as list
print('\nconvert edge pairs to id...')
pairs_id=[(node_names.index(i[0]),node_names.index(i[1])) for i in pairs_uniq]
print('#pairs id: ',len(pairs_id))

# Initialise adjacency matrix (all compornents are zero)
print('\ninitialise 31003*31003 adjacency matrix filled with zero...')
adj=np.zeros((31003,31003),dtype='int')

# convert edge positive component from '0' to '1'
print('\nconvert edge positive compornent to 1...')
for i in pairs_id:
	adj[i[0],i[1]]=1

# save ajd
savename='adj.npy'
print('\nsave adjacency matrix as {0}...'.format(savename))
np.save(savename,adj)

elapsed_time=time.time() - start_time
print('\n#time:{0}'.format(elapsed_time)+' sec')

print('-- fin --\n')

