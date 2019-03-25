from keras.models import Sequential,Model
from keras.layers import Dense
import networkx as nx
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from keras import backend as K, regularizers
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import math
import networkx as nx
import numpy as np
from functools import reduce

import keras
from keras import backend as K, regularizers
from keras.models import Model
from keras.layers import Dense, Embedding, Input, Reshape, Lambda

# Loss function
def build_reconstruction_loss(beta):
    """
    return the loss function for 2nd order proximity

    beta: the definition below Equation 3"""
    assert beta > 1

    def reconstruction_loss(true_y, pred_y):
        diff = K.square(true_y - pred_y)

        # borrowed from https://github.com/suanrong/SDNE/blob/master/model/sdne.py#L93
        weight = true_y * (beta - 1) + 1

        weighted_diff = diff * weight
        return K.mean(K.sum(weighted_diff, axis=1))  # mean square error
    return reconstruction_loss


def edge_wise_loss(true_y, embedding_diff):
    """1st order proximity
    """
    # true_y supposed to be None
    # we don't use it
    return K.mean(K.sum(K.square(embedding_diff), axis=1))  # mean square error


g = nx.read_edgelist('dataset/karate.edgelist', create_using=nx.Graph())
g = nx.convert_node_labels_to_integers(g)
graph = g
N = graph.number_of_nodes()
adj_mat = nx.adjacency_matrix(graph).toarray()


embedding_dim = 9
encode_dims = [34,20,16]
decode_dims = [16,20,34]

beta=2
alpha=2
l2_param=1e-3

encoding_layer_dims = encode_dims
decoding_layer_dims = decode_dims

#edges = np.array(list(self.graph.edges_iter()))

print(adj_mat)

#SDNE Model
model = Sequential()

# Creating Encoding Layers According to Requirement
for i, dim in enumerate(encoding_layer_dims):
            layer = Dense(dim, kernel_initializer='normal', activation='sigmoid',kernel_regularizer=regularizers.l2(l2_param), name='encoding-layer-{}'.format(i))
            model.add(layer)


model.add(Dense(embedding_dim, kernel_initializer='normal', activation='sigmoid', name='encoder'))


#Creating Decoding Layers Accordinng to Requirement
for i, dim in enumerate(decoding_layer_dims):
            layer = Dense(dim, kernel_initializer='normal', activation='sigmoid',kernel_regularizer=regularizers.l2(l2_param), name='decoding-layer-{}'.format(i))
            model.add(layer)

reconstruction_loss_function = build_reconstruction_loss(beta)

model.compile(loss=reconstruction_loss_function, optimizer='adadelta', metrics=['accuracy'])
model.fit(adj_mat,adj_mat,epochs=100)

print("###############################################################################################")

#Encoder Model
encoder=Model(model.input, model.get_layer('encoder').output)

#print(encoder.predict(adj_mat))
#Save the embedding
embedding = encoder.predict(adj_mat)
np.savetxt('karate.embedding2',embedding)

