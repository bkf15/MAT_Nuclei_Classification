import numpy as np
import json
import networkx as nx				#graphs need to be networkX graphs
from karateclub import Graph2Vec

#load dataset
txt = open('brecahad_nuclei_graph_data.json', 'r').read()
data = json.loads(txt)

#print(data[1]["X"])		#Accessing each field, "A", "X", and "y", for an individual sample

#need graph data to be in a list of networkX graphs, and labels to be np array
graph_list = []
labels = []
#print(labels)
for i in range(len(data)):
	#need to define nx graph structure, then do set_node_attributes for features
	g = nx.from_numpy_matrix(np.array(data[i]["A"]))
	feats = np.array(data[i]["X"])		#features for the nodes [Ribbon Taper Separation]
	#need to convert to dict of dict, keyed by node id's 
	feat_dict = {}
	for j in range(len(feats)):
		feat_dict.update({j: {'Ribbon' : feats[j][0], 'Taper' : feats[j][1], 'Separation' : feats[j][2]}})
	nx.set_node_attributes(g, feat_dict)
	graph_list.append(g)
	labels.append(data[i]["y"])

#convert label array to np array		
labels = np.array(labels)

######################model#######################
model = Graph2Vec(dimensions=10, seed=0, attributed=True)
model.fit(graph_list)		
embeddings = model.get_embedding()		#128 dimensions per nuclei default

print(embeddings[0])

print('done')