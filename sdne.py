import math
import networkx as nx
import numpy as np
from functools import reduce

import keras
from keras import backend as K, regularizers
from keras.models import Model
from keras.layers import Dense, Embedding, Input, Reshape, Lambda

class SNDE:
    def __init__(self,
                 graph,
                 encode_dim,
                 weight='weight',
                 encoding_layer_dims=[],
                 beta=2,
                 alpha=2,
                 l2_param=1e-3):
        
        # Graph Setting Up
        self.encode_dim = encode_dim
        self.graph = graph
        self.N = graph.number_of_nodes()
        self.adj_mat = nx.adjacency_matrix(self.graph).toarray()
        self.edges = np.array(list(self.graph.edges_iter()))

        # Weights Default to 1 
        weights = [graph[u][v].get(weight, 1.0)
                   for u, v in self.graph.edges_iter()]
        self.weights = np.array(weights, dtype=np.float32)[:, None]

        if len(self.weights) == self.weights.sum():
            print('the graph is unweighted')
        

        # Get Started
        # 16 10
        model = Sequential()
        for dim in encoding_layer_dims:
            model.add(Dense(dim, activation='sigmoid')
        
        # 8
        model.add(Dense(encode_dim, activation='sigmoid'))
        
        # 10 16         
        #Decoding
        decoding_layer_dims = encoding_layer_dims[::-1]
        for dim in decoding_layer_dims:
            model.add(Dense(dim, activation='sigmoid'))

