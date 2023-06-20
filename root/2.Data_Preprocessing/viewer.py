import numpy as np 
import warnings
import networkx as nx
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score
import copy

a = 'Data Tables Step 7 - Processed npzs/graph_0_general_attributes.npz'

with np.load(a, allow_pickle=True) as loader:
    loader = dict(loader)
    A = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                loader['adj_indptr']), shape=loader['adj_shape'])

    g = nx.from_scipy_sparse_matrix(A, create_using=nx.DiGraph())
    print("k")
