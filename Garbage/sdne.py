import math
import networkx as nx
import numpy as np
from functools import reduce

import keras
from keras import backend as K, regularizers
from keras.models import Model
from keras.layers import Dense, Embedding, Input, Reshape, Lambda

class SNDE:
    __init__(self,graph,encode_dim, weight='weight', encoding_layer_dims=[] , beta=2, alpha=2, l2_param=1e-3):


