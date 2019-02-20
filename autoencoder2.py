import numpy

from keras.models import Sequential,Model
from keras.layers import Dense
import networkx as nx
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder


g = nx.read_edgelist('dataset/g.edgelist', create_using=nx.Graph())
g = nx.convert_node_labels_to_integers(g)
graph = g
N = graph.number_of_nodes()
adj_mat = nx.adjacency_matrix(graph).toarray()

#edges = np.array(list(self.graph.edges_iter()))

#print(adj_mat)

#SDNE Model
model = Sequential()
model.add(Dense(5242, input_dim=5242, kernel_initializer='normal', activation='sigmoid'))
model.add(Dense(2500, kernel_initializer='normal', activation='sigmoid'))
model.add(Dense(500, kernel_initializer='normal', activation='sigmoid'))
model.add(Dense(100, kernel_initializer='normal', activation='sigmoid', name='encoder'))
model.add(Dense(500, kernel_initializer='normal', activation='sigmoid'))
model.add(Dense(2500, kernel_initializer='normal', activation='sigmoid'))
model.add(Dense(5242, kernel_initializer='normal', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
#model.compile(loss='poisson', optimizer='adadelta', metrics=['accuracy'])
model.fit(adj_mat,adj_mat,epochs=2)

print("###############################################################################################")

#Encoder Model
encoder=Model(model.input, model.get_layer('encoder').output)

print(encoder.predict(adj_mat))
