#FIRST MAKE THE CONCATENATED GRAPH CSV SPARSE -> ENSURE THAT NO SELF LOOP AND THE GRAPH IS UNDIRECTED

import numpy as np
import scipy.sparse as sp 
import pandas as pd
import networkx as nx 
from networkx import from_scipy_sparse_matrix
import networkx
from sklearn import preprocessing
import copy


def edges_to_sparse(edges, N, values=None):
    if values is None:
        values = np.ones(edges.shape[0])
    
    return sp.coo_matrix((values, (edges[:, 0], edges[:, 1])), shape=(N, N)).tocsr()

def check_graph(sp_graph):
    A = sp_graph
    is_undirected = (A != A.T).nnz == 0
    print(is_undirected)

    print(A.diagonal().sum())

np.set_printoptions(suppress=True)
#graph_file = 'data/graph_concatenated.csv'
graph_folder = 'Data Tables Step 5 - Graph/'
attribute_folder = 'Data Tables Step 6 - Attributes/'
final_folder = "Data Tables Step 7 - Processed npzs_/"

graph_house_region = pd.read_csv(graph_folder + 'house-region_modified.csv', dtype = 'float32')
graph_house_train = pd.read_csv(graph_folder + 'house-train.csv', dtype = 'float32')
graph_property_school = pd.read_csv(graph_folder + 'property-school.csv', dtype = 'float32')
graph_school_train = pd.read_csv(graph_folder + 'school-train.csv', dtype = 'float32')
graph_train_train = pd.read_csv(graph_folder + 'train-train.csv', dtype = 'float32')
graph_all = np.vstack((graph_house_region.values, graph_house_train.values, graph_property_school.values, 
                        graph_school_train.values, graph_train_train.values))

sp_val_house_region = edges_to_sparse(graph_house_region.values[:, 0:2], 67117, graph_house_region.values[:, 2])
sp_val_house_train = edges_to_sparse(graph_house_train.values[:, 0:2], 67117, graph_house_train.values[:, 2])
sp_val_property_school = edges_to_sparse(graph_property_school.values[:, 0:2], 67117, graph_property_school.values[:, 2])
sp_val_school_train = edges_to_sparse(graph_school_train.values[:, 0:2], 67117, graph_school_train.values[:, 2])
sp_val_train_train = edges_to_sparse(graph_train_train.values[:, 0:2], 67117, graph_train_train.values[:, 2])
 #DIDNOT USE .VALUES AS IT'S ALREADY NUMPY
sp_val_graph_all = edges_to_sparse(graph_all[:, 0:2], 67117, graph_all[:, 2])

check_graph(sp_val_house_region)
check_graph(sp_val_house_train)
check_graph(sp_val_property_school)
check_graph(sp_val_school_train)
check_graph(sp_val_train_train)
check_graph(sp_val_graph_all)

nx_graph_house_region = from_scipy_sparse_matrix(sp_val_house_region, create_using=networkx.DiGraph())
#nx_graph_house_region = from_scipy_sparse_matrix(sp_val_house_region)
#nx_graph_house_region_modified = nx_graph_house_region.to_undirected()
#print(type(nx_graph_modified))
#A1 = nx.to_scipy_sparse_matrix(nx_graph_house_region_modified)
A1 = nx.to_scipy_sparse_matrix(nx_graph_house_region)

check_graph(A1)

nx_graph_house_train = from_scipy_sparse_matrix(sp_val_house_train, create_using=networkx.DiGraph())
#nx_graph_house_train = from_scipy_sparse_matrix(sp_val_house_train)
#nx_graph_house_train_modified = nx_graph_house_train.to_undirected()
#print(type(nx_graph_modified))
#A2 = nx.to_scipy_sparse_matrix(nx_graph_house_train_modified)
A2 = nx.to_scipy_sparse_matrix(nx_graph_house_train)

check_graph(A2)

nx_graph_property_school = from_scipy_sparse_matrix(sp_val_property_school, create_using=networkx.DiGraph())
#nx_graph_property_school = from_scipy_sparse_matrix(sp_val_property_school)
#nx_graph_property_school_modified = nx_graph_property_school.to_undirected()
#print(type(nx_graph_modified))
#A3 = nx.to_scipy_sparse_matrix(nx_graph_property_school_modified)
A3 = nx.to_scipy_sparse_matrix(nx_graph_property_school)

check_graph(A3)

nx_graph_school_train = from_scipy_sparse_matrix(sp_val_school_train, create_using=networkx.DiGraph())
#nx_graph_school_train = from_scipy_sparse_matrix(sp_val_school_train)
#nx_graph_school_train_modified = nx_graph_school_train.to_undirected()
#print(type(nx_graph_modified))
#A4 = nx.to_scipy_sparse_matrix(nx_graph_school_train_modified)
A4 = nx.to_scipy_sparse_matrix(nx_graph_school_train)

check_graph(A4)

nx_graph_train_train = from_scipy_sparse_matrix(sp_val_train_train, create_using=networkx.DiGraph())
#nx_graph_train_train = from_scipy_sparse_matrix(sp_val_train_train)
#nx_graph_train_train_modified = nx_graph_train_train.to_undirected()
#print(type(nx_graph_modified))
#A5 = nx.to_scipy_sparse_matrix(nx_graph_train_train_modified)
A5 = nx.to_scipy_sparse_matrix(nx_graph_train_train)
check_graph(A5)

nx_graph_all = from_scipy_sparse_matrix(sp_val_graph_all)
nx_graph_all2 = nx_graph_all.to_directed()
A0 = nx.to_scipy_sparse_matrix(nx_graph_all)
check_graph(A0)

#########################################################################

#ATTRIBUTE HANDLING.
#STEPS: i) SEPARATE LABELS OUT /
# ii) CREATE A NEW FILE, ID AND PRICE /
# iii) FULL ZERO MEAN, UNIT VARIANCE CONVERSION

TRAIN_ID_START = 0
TRAIN_ID_END = 218
""" REGION_ID_START = 218
REGION_ID_END = 13557
SCHOOL_ID_START = 13557
SCHOOL_ID_END = 14266
PROPERTY_ID_START = 14266
PROPERTY_ID_END = 67118 """
#PROPERTY_ID_START = 218
#PROPERTY_ID_END = None


attr_property = pd.read_csv(attribute_folder + 'property.csv')
#ATTEMPTING REPLACING OUTLIERS WITH 3,-3
attr_property_temp = attr_property.iloc[:, 1:]
attr_property_temp[ (attr_property_temp) > 3 ] = 3
attr_property_temp[ (attr_property_temp) < -3 ] = -3
tmp_nparr = attr_property.values
tmp_nparr[:, 1:] = attr_property_temp.values
temparr = np.zeros((67117, len(tmp_nparr[0])))
temparr[int(tmp_nparr[0,0]):int(tmp_nparr[-1, 0]+1)] = tmp_nparr
#temparr[int(tmp_nparr[0,0]):int(tmp_nparr[-1, 0]+1)] = attr_property_temp.values
attr_property_sparse = sp.csr_matrix(temparr[:, 1:])

attr_region = pd.read_csv(attribute_folder + 'region_modified.csv').drop(['Lat','Lng'], axis=1)
#ATTEMPTING REPLACING OUTLIERS WITH 3,-3
attr_region_temp = attr_region.iloc[:, 1:]
attr_region_temp[ (attr_region_temp) > 3 ] = 3
attr_region_temp[ (attr_region_temp) < -3 ] = -3
tmp_nparr = attr_region.values
tmp_nparr[:, 1:] = attr_region_temp.values
temparr = np.zeros((67117, len(tmp_nparr[0])))
temparr[int(tmp_nparr[0,0]):int(tmp_nparr[-1, 0]+1)] = tmp_nparr
#temparr[int(tmp_nparr[0,0]):int(tmp_nparr[-1, 0]+1)] = attr_region_temp.values
attr_region_sparse = sp.csr_matrix(temparr[:, 1:])

attr_school = pd.read_csv(attribute_folder + 'school.csv')
#ATTEMPTING REPLACING OUTLIERS WITH 3,-3
attr_school_temp = attr_school.iloc[:, 1:]
attr_school_temp[ (attr_school_temp) > 3 ] = 3
attr_school_temp[ (attr_school_temp) < -3 ] = -3
tmp_nparr = attr_school.values
tmp_nparr[:, 1:] = attr_school_temp.values
temparr = np.zeros((67117, len(tmp_nparr[0])))
temparr[int(tmp_nparr[0,0]):int(tmp_nparr[-1, 0]+1)] = tmp_nparr
#temparr[int(tmp_nparr[0,0]):int(tmp_nparr[-1, 0]+1)] = attr_school_temp.values
attr_school_sparse = sp.csr_matrix(temparr[:, 1:])

attr_train = pd.read_csv(attribute_folder + 'train.csv')
#ATTEMPTING REPLACING OUTLIERS WITH 3,-3
attr_train_temp = attr_train.iloc[:, 1:]
attr_train_temp[ (attr_train_temp) > 3 ] = 3
attr_train_temp[ (attr_train_temp) < -3 ] = -3
tmp_nparr = attr_train.values
tmp_nparr[:, 1:] = attr_train_temp.values
temparr = np.zeros((67117, len(tmp_nparr[0])))
temparr[int(tmp_nparr[0,0]):int(tmp_nparr[-1, 0]+1)] = tmp_nparr
#temparr[int(tmp_nparr[0,0]):int(tmp_nparr[-1, 0]+1)] = attr_train_temp.values
attr_train_sparse = sp.csr_matrix(temparr[:, 1:])

#attr = attr.values
#temp = copy.deepcopy(attr[:, 7:9])
#temp = preprocessing.scale(temp) #scaling the lat, lng attributes

#train normalize
#print(attr[TRAIN_ID_START: TRAIN_ID_END+1, :])
#attr[TRAIN_ID_START: TRAIN_ID_END, :] = preprocessing.scale(attr[TRAIN_ID_START: TRAIN_ID_END, :])
#attr[REGION_ID_START: REGION_ID_END, :] = preprocessing.scale(attr[REGION_ID_START: REGION_ID_END, :])
#attr[SCHOOL_ID_START: SCHOOL_ID_END, :] = preprocessing.scale(attr[SCHOOL_ID_START: SCHOOL_ID_END, :])
#attr[PROPERTY_ID_START: , :] = preprocessing.scale(attr[PROPERTY_ID_START: , :])
#attr[:, 7:9] = temp

""" print(attr[:, 1:])
attr = attr[:, 1:]
#attr = preprocessing.scale(attr[:, 1:])
print(attr.shape)
attr_sparse = sp.csr_matrix(attr)
print(attr_sparse.indptr)
print(attr_sparse.indices)
print(attr_sparse.data)
np.savetxt('tempcsv.csv', attr) """

########################################################################

#SAVE AS A FILE, THIS IS IT
np.savez(final_folder + 'graph_1_house_region.npz', adj_data = A1.data, adj_indices = A1.indices, adj_indptr = A1.indptr, adj_shape = A1.shape,
        attr_data1 = None, attr_indices1 = None, attr_indptr1 = None, attr_shape1 = None,
        attr_data2 = None, attr_indices2 = None, attr_indptr2 = None, attr_shape2 = None,
        attr_data3 = None, attr_indices3 = None, attr_indptr3 = None, attr_shape3 = None,
        attr_data4 = None, attr_indices4 = None, attr_indptr4 = None, attr_shape4 = None
        )

np.savez(final_folder + 'graph_2_house_train.npz', adj_data = A2.data, adj_indices = A2.indices, adj_indptr = A2.indptr, adj_shape = A2.shape,
        attr_data1 = None, attr_indices1 = None, attr_indptr1 = None, attr_shape1 = None,
        attr_data2 = None, attr_indices2 = None, attr_indptr2 = None, attr_shape2 = None,
        attr_data3 = None, attr_indices3 = None, attr_indptr3 = None, attr_shape3 = None,
        attr_data4 = None, attr_indices4 = None, attr_indptr4 = None, attr_shape4 = None
        )

np.savez(final_folder + 'graph_3_property_school.npz', adj_data = A3.data, adj_indices = A3.indices, adj_indptr = A3.indptr, adj_shape = A3.shape,
        attr_data1 = None, attr_indices1 = None, attr_indptr1 = None, attr_shape1 = None,
        attr_data2 = None, attr_indices2 = None, attr_indptr2 = None, attr_shape2 = None,
        attr_data3 = None, attr_indices3 = None, attr_indptr3 = None, attr_shape3 = None,
        attr_data4 = None, attr_indices4 = None, attr_indptr4 = None, attr_shape4 = None
        )

np.savez(final_folder + 'graph_4_school_train.npz', adj_data = A4.data, adj_indices = A4.indices, adj_indptr = A4.indptr, adj_shape = A4.shape,
        attr_data1 = None, attr_indices1 = None, attr_indptr1 = None, attr_shape1 = None,
        attr_data2 = None, attr_indices2 = None, attr_indptr2 = None, attr_shape2 = None,
        attr_data3 = None, attr_indices3 = None, attr_indptr3 = None, attr_shape3 = None,
        attr_data4 = None, attr_indices4 = None, attr_indptr4 = None, attr_shape4 = None
        )

np.savez(final_folder + 'graph_5_train_train.npz', adj_data = A5.data, adj_indices = A5.indices, adj_indptr = A5.indptr, adj_shape = A5.shape,
        attr_data1 = None, attr_indices1 = None, attr_indptr1 = None, attr_shape1 = None,
        attr_data2 = None, attr_indices2 = None, attr_indptr2 = None, attr_shape2 = None,
        attr_data3 = None, attr_indices3 = None, attr_indptr3 = None, attr_shape3 = None,
        attr_data4 = None, attr_indices4 = None, attr_indptr4 = None, attr_shape4 = None
        )

np.savez(final_folder + 'graph_0_general_attributes.npz', adj_data = A0.data, adj_indices = A0.indices, adj_indptr = A0.indptr, adj_shape = A0.shape,
        attr_data1 = attr_train_sparse.data , attr_indices1 = attr_train_sparse.indices, attr_indptr1 = attr_train_sparse.indptr, attr_shape1 = attr_train_sparse.shape,
        attr_data2 = attr_region_sparse.data, attr_indices2 = attr_region_sparse.indices, attr_indptr2 = attr_region_sparse.indptr, attr_shape2 = attr_region_sparse.shape,
        attr_data3 = attr_school_sparse.data, attr_indices3 = attr_school_sparse.indices, attr_indptr3 = attr_school_sparse.indptr, attr_shape3 = attr_school_sparse.shape,
        attr_data4 = attr_property_sparse.data, attr_indices4 = attr_property_sparse.indices, attr_indptr4 = attr_property_sparse.indptr, attr_shape4 = attr_property_sparse.shape
        )
