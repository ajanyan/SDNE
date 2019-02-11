import gzip
import pickle as pkl
import numpy as np
import networkx as nx
import math
from sdne import SDNE
from sklearn.model_selection import train_test_split
from itertools import product
from tqdm import tqdm
from keras.callbacks import Callback

batch_size = 64
ks = [1, 5, 50, 100, 150, 250, 400, 500, 5000]

g = nx.read_edgelist('data/grqc.txt', create_using=nx.Graph())
g = nx.convert_node_labels_to_integers(g)

parameter_dicts = [{'beta': 2, 'alpha': 2, 'l2_param': 1e-4}]

dev_ratio = 0.1
test_ratio = 0.15

train_set, test_edges = train_test_split(g.edges(), test_size=test_ratio)
g.remove_edges_from(test_edges)
g.add_edges_from([(i,i) for i in np.arange(g.number_of_nodes())])

