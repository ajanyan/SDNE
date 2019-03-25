from keras.models import Sequential,Model
from keras.layers import Dense
import networkx as nx
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

g = nx.read_edgelist('dataset/karate.edgelist', create_using=nx.Graph())
g = nx.convert_node_labels_to_integers(g)
graph = g
N = graph.number_of_nodes()
adj_mat = nx.adjacency_matrix(graph).toarray()


embedding_dim = 9
encode_dims = [34,20,16]
decode_dims = [16,20,34]


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

model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.fit(adj_mat,adj_mat,epochs=1000)

print("###############################################################################################")

#Encoder Model
encoder=Model(model.input, model.get_layer('encoder').output)

#print(encoder.predict(adj_mat))
#Save the embedding
embedding = encoder.predict(adj_mat)
np.savetxt('karate.embedding2',embedding)

