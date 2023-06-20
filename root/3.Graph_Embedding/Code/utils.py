
import warnings
import networkx as nx
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score
import copy

node1_start, node1_end = (0, 218)
node2_start, node2_end = (218, 13557)
node3_start, node3_end = (13557, 14266)
node4_start, node4_end = (14266, 67117)

node_trackers = []
node_trackers.append((0, 218))
node_trackers.append((218, 13557))
node_trackers.append((13557, 14266))
node_trackers.append((14266, 67117))


def determine_type(node_no):
        if node_no >= node1_start and node_no < node1_end:
            return 0
        elif node_no >= node2_start and node_no < node2_end:
            return 1
        elif node_no >= node3_start and node_no < node3_end:
            return 2
        else:
            return 3


class DataUtils:
    def __init__(self, graph_file, is_all=False, node_negative_distribution_temp = None, test_indices = None):
        self.test_indices = test_indices

        with np.load(graph_file, allow_pickle=True) as loader:
            loader = dict(loader)
            self.A = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                               loader['adj_indptr']), shape=loader['adj_shape'])
            
            if loader['attr_data1'].all() != None:
                print("Attributes Found.")

                self.X1 = sp.csr_matrix((loader['attr_data1'], loader['attr_indices1'],
                                loader['attr_indptr1']), shape=loader['attr_shape1'])
                self.X2 = sp.csr_matrix((loader['attr_data2'], loader['attr_indices2'],
                                loader['attr_indptr2']), shape=loader['attr_shape2'])
                self.X3 = sp.csr_matrix((loader['attr_data3'], loader['attr_indices3'],
                                loader['attr_indptr3']), shape=loader['attr_shape3'])
                self.X4 = sp.csr_matrix((loader['attr_data4'], loader['attr_indices4'],
                                loader['attr_indptr4']), shape=loader['attr_shape4'])

            """ self.node_type1 = loader['node_type1']
            self.node_type2 = loader['node_type2'] """

            if 'labels' in loader.keys():
                self.labels = loader['labels']
            else:
                self.labels = None

            if not is_all and 'val_edges' in loader.keys():
                raise Exception("val not included yet")
                """ self.val_edges = loader['val_edges']
                self.val_ground_truth = loader['val_ground_truth']
                self.test_edges = loader['test_edges']
                self.test_ground_truth = loader['test_ground_truth'] """

            self.g = nx.from_scipy_sparse_matrix(self.A, create_using=nx.DiGraph())
            if type(self.test_indices) != type(None):
                self.g.remove_edges_from(list(self.g.in_edges(test_indices)))
                self.g.remove_edges_from(list(self.g.out_edges(test_indices)))

            self.num_of_nodes = self.g.number_of_nodes()
            self.num_of_edges = self.g.number_of_edges()
            self.edges_raw = self.g.edges(data=True)
            self.nodes_raw = self.g.nodes(data=True)

            #self.edge_distribution = np.array([attr['weight'] for _, _, attr in self.edges_raw], dtype=np.float32)
            self.edge_distribution = np.array([1/attr['weight'] if attr['weight']>0 else 0 for _, _, attr in self.edges_raw], dtype=np.float32)
            self.edge_distribution /= np.sum(self.edge_distribution)
            self.edge_sampling = AliasSampling(prob=self.edge_distribution)
            ''' self.node_negative_distribution = np.power(
                np.array([self.g.degree(node, weight='weight') for node, _ in self.nodes_raw], dtype=np.float32), 0.75) '''

            if type(node_negative_distribution_temp) != type(None):
                self.node_negative_distribution_temp = node_negative_distribution_temp
                #HERE WE HAVE TO BUILD TWO DIFFERENT NODE NEGATIVE SAMPLER
                sample_node0, sample_node1, _ = list(self.edges_raw)[0]
                node_type1, node_type2 = determine_type(sample_node0), determine_type(sample_node1)
                #print("node types: "+str(node_type1)+" "+str(node_type2))

                node_type1_start, node_type1_end = node_trackers[node_type1]
                node_type2_start, node_type2_end = node_trackers[node_type2]

                self.node_negative_distribution_temp_type1 = copy.deepcopy(self.node_negative_distribution_temp)
                self.node_negative_distribution_temp_type1[0: node_type1_start] = 0
                self.node_negative_distribution_temp_type1[node_type1_end: ] = 0
                print("sum 1 "+str(np.sum(self.node_negative_distribution_temp_type1)))
                
                if(np.sum(self.node_negative_distribution_temp_type1) == 0):
                  print("ZERO DIVIDE WARNING!!! ")
                self.node_negative_distribution_type1 = self.node_negative_distribution_temp_type1/np.sum(self.node_negative_distribution_temp_type1)
                print(self.node_negative_distribution_temp_type1)
                self.node_sampling_type1 = AliasSampling(prob=self.node_negative_distribution_type1)

                self.node_negative_distribution_temp_type2 = copy.deepcopy(self.node_negative_distribution_temp)
                self.node_negative_distribution_temp_type2[0: node_type2_start] = 0
                self.node_negative_distribution_temp_type2[node_type2_end: ] = 0
                print("sum 2 "+str(np.sum(self.node_negative_distribution_temp_type2)))

                self.node_negative_distribution_type2 = self.node_negative_distribution_temp_type2/np.sum(self.node_negative_distribution_temp_type2)
                self.node_sampling_type2 = AliasSampling(prob=self.node_negative_distribution_type2)

            else:
                print("Calculating global graph node properties...")
                g_temp = self.g.to_undirected().to_directed()
                print(g_temp.degree(0))
                self.node_negative_distribution_temp = np.power(
                    np.array([1/g_temp.degree(node, weight='weight') if g_temp.degree(node, weight='weight')>0 else 0 for node, _ in self.nodes_raw], dtype=np.float32), 0.75)
                ''' print(self.node_negative_distribution_temp[0:218])
                print("**********************")
                print(self.node_negative_distribution_temp[218:13557])
                print("**********************")
                print(self.node_negative_distribution_temp[13557:14266])
                print("**********************")
                print(self.node_negative_distribution_temp[14266:])
                print("++++++++++++++++++++++++++++++++++++++++++++++=====================")
 '''
                self.node_negative_distribution = self.node_negative_distribution_temp/np.sum(self.node_negative_distribution_temp)
                self.node_sampling = AliasSampling(prob=self.node_negative_distribution)

            ''' if node_negative_distribution_temp.all() == None:
                print("Calculating global graph node properties...")
                self.node_negative_distribution_temp = np.power(
                    np.array([1/self.g.degree(node, weight='weight') if self.g.degree(node, weight='weight')>0 else 0 for node, _ in self.nodes_raw], dtype=np.float32), 0.75)

                self.node_negative_distribution = self.node_negative_distribution_temp/np.sum(self.node_negative_distribution_temp)
                self.node_sampling = AliasSampling(prob=self.node_negative_distribution)
                
            else:
                self.node_negative_distribution_temp = node_negative_distribution_temp
                #HERE WE HAVE TO BUILD TWO DIFFERENT NODE NEGATIVE SAMPLER
                sample_node0, sample_node1, _ = self.edges_raw[0]
                node_type1, node_type2 = determine_type(sample_node0), determine_type(sample_node1)

                node_type1_start, node_type1_end = node_trackers[node_type1]
                node_type2_start, node_type2_end = node_trackers[node_type2]

                self.node_negative_distribution_temp_type1 = copy.deepcopy(self.node_negative_distribution_temp)
                self.node_negative_distribution_temp_type1[0: node_type1_start] = 0
                self.node_negative_distribution_temp_type1[node_type1_end: ] = 0
                self.node_negative_distribution_type1 = self.node_negative_distribution_temp_type1/np.sum(self.node_negative_distribution_temp_type1)
                self.node_sampling_type1 = AliasSampling(prob=self.node_negative_distribution_type1)

                self.node_negative_distribution_temp_type2 = copy.deepcopy(self.node_negative_distribution_temp)
                self.node_negative_distribution_temp_type2[0: node_type2_start] = 0
                self.node_negative_distribution_temp_type2[node_type2_end: ] = 0
                self.node_negative_distribution_type2 = self.node_negative_distribution_temp_type2/np.sum(self.node_negative_distribution_temp_type2)
                self.node_sampling_type2 = AliasSampling(prob=self.node_negative_distribution_type2) '''

            self.node_index = {}
            self.node_index_reversed = {}
            for index, (node, _) in enumerate(self.nodes_raw):
                if index != node:
                  raise Exception("Discrepancy!!!!")
                self.node_index[node] = index
                self.node_index_reversed[index] = node
            self.edges = [(self.node_index[u], self.node_index[v]) for u, v, _ in self.edges_raw]


    def fetch_next_batch(self, batch_size=16, K=5):
        
        edge_batch_index = self.edge_sampling.sampling(batch_size)
        
        u_i = []
        u_j = []
        label = []
        reverse = False
        if np.random.rand() > 0.5:
            reverse = True
        for edge_index in edge_batch_index:
            edge = self.edges[edge_index]
            #if self.g.__class__ == nx.Graph:
            #    if reverse == True:
            #        edge = (edge[1], edge[0])
            if reverse == True:
              edge = (edge[1], edge[0])
            u_i.append(edge[0])
            u_j.append(edge[1])
            label.append(1)
            node_type_1 = determine_type(edge[0])
            node_type_2 = determine_type(edge[1])

            for i in range(K):
                while True:
                    negative_node = self.node_sampling_type2.sampling() if reverse == False else self.node_sampling_type1.sampling()
                    if determine_type(negative_node) == node_type_2 and (not self.g.has_edge(self.node_index_reversed[negative_node], self.node_index_reversed[edge[0]])):
                        break
                    """ elif determine_type(negative_node) != node_type_2:
                      raise Exception("Problem in node sampling causing processing delay") """
                      #print("neg type: " + str(determine_type(negative_node)))
                      #print("pos type: " + str(node_type_2))

                u_i.append(edge[0])
                u_j.append(negative_node)
                label.append(-1)

        '''if len(np.intersect1d(u_i, self.test_indices)) + len(np.intersect1d(u_j, self.test_indices)) > 0:
            raise Exception("Error ensuring test set not separate from training. Inductivity couldn't be confirmed.")'''
        
        return u_i, u_j, label, node_type_1, node_type_2

    def embedding_mapping(self, embedding):
        return {node: embedding[self.node_index[node]] for node, _ in self.nodes_raw}


class AliasSampling:
    # Reference: LINE source code from https://github.com/snowkylin/line
    # Reference: https://en.wikipedia.org/wiki/Alias_method
    def __init__(self, prob):
        self.n = len(prob)
        self.U = np.array(prob) * self.n
        self.K = [i for i in range(len(prob))]
        overfull, underfull = [], []
        for i, U_i in enumerate(self.U):
            if U_i > 1:
                overfull.append(i)
            elif U_i < 1:
                underfull.append(i)
        while len(overfull) and len(underfull):
            i, j = overfull.pop(), underfull.pop()
            self.K[j] = i
            self.U[i] = self.U[i] - (1 - self.U[j])
            if self.U[i] > 1:
                overfull.append(i)
            elif self.U[i] < 1:
                underfull.append(i)

    def sampling(self, n=1):
        x = np.random.rand(n)
        i = np.floor(self.n * x)
        y = self.n * x - i
        i = i.astype(np.int32)
        res = [i[k] if y[k] < self.U[i[k]] else self.K[i[k]] for k in range(n)]
        if n == 1:
            return res[0]
        else:
            return res


def train_val_test_split(graph_file, p_test=0.10, p_val=0.05):
    with np.load(graph_file, allow_pickle=True) as loader:
        loader = dict(loader)
        A = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                           loader['adj_indptr']), shape=loader['adj_shape'])

        X = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                           loader['attr_indptr']), shape=loader['attr_shape'])

        if 'labels' in loader.keys():
            labels = loader['labels']
        else:
            labels = None

        train_ones, val_ones, val_zeros, test_ones, test_zeros = _train_val_test_split_adjacency(A=A, p_test=p_test, p_val=p_val, neg_mul=1, every_node=True, connected=False, undirected=(A != A.T).nnz == 0)
        if p_val > 0:
            val_edges = np.row_stack((val_ones, val_zeros))
            val_ground_truth = A[val_edges[:, 0], val_edges[:, 1]].A1
            val_ground_truth = np.where(val_ground_truth > 0, 1, val_ground_truth)
        if p_test > 0:
            test_edges = np.row_stack((test_ones, test_zeros))
            test_ground_truth = A[test_edges[:, 0], test_edges[:, 1]].A1
            test_ground_truth = np.where(test_ground_truth > 0, 1, test_ground_truth)
            if p_val == 0:
                val_edges = test_edges
                val_ground_truth = test_ground_truth
        A = edges_to_sparse(train_ones, A.shape[0])
    return A, X, labels, val_edges, val_ground_truth, test_edges, test_ground_truth


def _train_val_test_split_adjacency(A, p_val=0.10, p_test=0.05, seed=0, neg_mul=1,
                                    every_node=True, connected=False, undirected=False,
                                    use_edge_cover=True, set_ops=True, asserts=False):
    # Reference: G2G source code from https://github.com/abojchevski/graph2gauss
    assert p_val + p_test > 0
    assert A.min() == 0  # no negative edges
    assert A.diagonal().sum() == 0  # no self-loops
    assert not np.any(A.sum(0).A1 + A.sum(1).A1 == 0)  # no dangling nodes
    is_undirected = (A != A.T).nnz == 0

    if undirected:
        assert is_undirected  # make sure is directed
        A = sp.tril(A).tocsr()  #    consider only upper triangular
        A.eliminate_zeros()
    else:
        if is_undirected:
            warnings.warn('Graph appears to be undirected. Did you forgot to set undirected=True?')

    np.random.seed(seed)

    E = A.nnz
    N = A.shape[0]
    s_train = int(E * (1 - p_val - p_test))

    idx = np.arange(N)

    # hold some edges so each node appears at least once
    if every_node:
        if connected:
            assert sp.csgraph.connected_components(A)[0] == 1  # make sure original graph is connected
            A_hold = sp.csgraph.minimum_spanning_tree(A)
        else:
            A.eliminate_zeros()  # makes sure A.tolil().rows contains only indices of non-zero elements
            d = A.sum(1).A1

            if use_edge_cover:
                hold_edges = edge_cover(A)

                # make sure the training percentage is not smaller than len(edge_cover)/E when every_node is set to True
                min_size = hold_edges.shape[0]
                if min_size > s_train:
                    raise ValueError('Training percentage too low to guarantee every node. Min train size needed {:.2f}'
                                     .format(min_size / E))
            else:
                # make sure the training percentage is not smaller than N/E when every_node is set to True
                if N > s_train:
                    raise ValueError('Training percentage too low to guarantee every node. Min train size needed {:.2f}'
                                     .format(N / E))

                hold_edges_d1 = np.column_stack(
                    (idx[d > 0], np.row_stack(map(np.random.choice, A[d > 0].tolil().rows))))

                if np.any(d == 0):
                    hold_edges_d0 = np.column_stack((np.row_stack(map(np.random.choice, A[:, d == 0].T.tolil().rows)),
                                                     idx[d == 0]))
                    hold_edges = np.row_stack((hold_edges_d0, hold_edges_d1))
                else:
                    hold_edges = hold_edges_d1

            if asserts:
                assert np.all(A[hold_edges[:, 0], hold_edges[:, 1]])
                assert len(np.unique(hold_edges.flatten())) == N

            A_hold = edges_to_sparse(hold_edges, N)

        A_hold[A_hold > 1] = 1
        A_hold.eliminate_zeros()
        A_sample = A - A_hold

        s_train = s_train - A_hold.nnz
    else:
        A_sample = A

    idx_ones = np.random.permutation(A_sample.nnz)
    ones = np.column_stack(A_sample.nonzero())
    train_ones = ones[idx_ones[:s_train]]
    test_ones = ones[idx_ones[s_train:]]

    # return back the held edges
    if every_node:
        train_ones = np.row_stack((train_ones, np.column_stack(A_hold.nonzero())))

    n_test = len(test_ones) * neg_mul
    if set_ops:
        # generate slightly more completely random non-edge indices than needed and discard any that hit an edge
        # much faster compared a while loop
        # in the future: estimate the multiplicity (currently fixed 1.3/2.3) based on A_obs.nnz
        if undirected:
            random_sample = np.random.randint(0, N, [int(2.3 * n_test), 2])
            random_sample = random_sample[random_sample[:, 0] > random_sample[:, 1]] #only upper triangle
        else:
            random_sample = np.random.randint(0, N, [int(1.3 * n_test), 2])
            random_sample = random_sample[random_sample[:, 0] != random_sample[:, 1]]

        # discard ones
        random_sample = random_sample[A[random_sample[:, 0], random_sample[:, 1]].A1 == 0]
        # discard duplicates
        random_sample = random_sample[np.unique(random_sample[:, 0] * N + random_sample[:, 1], return_index=True)[1]]
        # only take as much as needed
        test_zeros = np.row_stack(random_sample)[:n_test]
        assert test_zeros.shape[0] == n_test
    else:
        test_zeros = []
        while len(test_zeros) < n_test:
            i, j = np.random.randint(0, N, 2)
            if A[i, j] == 0 and (not undirected or i > j) and (i, j) not in test_zeros:
                test_zeros.append((i, j))
        test_zeros = np.array(test_zeros)

    # split the test set into validation and test set
    s_val_ones = int(len(test_ones) * p_val / (p_val + p_test))
    s_val_zeros = int(len(test_zeros) * p_val / (p_val + p_test))

    val_ones = test_ones[:s_val_ones]
    test_ones = test_ones[s_val_ones:]

    val_zeros = test_zeros[:s_val_zeros]
    test_zeros = test_zeros[s_val_zeros:]

    if undirected:
        # put (j, i) edges for every (i, j) edge in the respective sets and form back original A
        symmetrize = lambda x: np.row_stack((x, np.column_stack((x[:, 1], x[:, 0]))))
        train_ones = symmetrize(train_ones)
        val_ones = symmetrize(val_ones)
        val_zeros = symmetrize(val_zeros)
        test_ones = symmetrize(test_ones)
        test_zeros = symmetrize(test_zeros)
        A = A.maximum(A.T)

    if asserts:
        set_of_train_ones = set(map(tuple, train_ones))
        assert train_ones.shape[0] + test_ones.shape[0] + val_ones.shape[0] == A.nnz
        assert (edges_to_sparse(np.row_stack((train_ones, test_ones, val_ones)), N) != A).nnz == 0
        assert set_of_train_ones.intersection(set(map(tuple, test_ones))) == set()
        assert set_of_train_ones.intersection(set(map(tuple, val_ones))) == set()
        assert set_of_train_ones.intersection(set(map(tuple, test_zeros))) == set()
        assert set_of_train_ones.intersection(set(map(tuple, val_zeros))) == set()
        assert len(set(map(tuple, test_zeros))) == len(test_ones) * neg_mul
        assert len(set(map(tuple, val_zeros))) == len(val_ones) * neg_mul
        assert not connected or sp.csgraph.connected_components(A_hold)[0] == 1
        assert not every_node or ((A_hold - A) > 0).sum() == 0

    return train_ones, val_ones, val_zeros, test_ones, test_zeros


def edge_cover(A):
    # Reference: G2G source code from https://github.com/abojchevski/graph2gauss
    N = A.shape[0]
    d_in = A.sum(0).A1
    d_out = A.sum(1).A1

    # make sure to include singleton nodes (nodes with one incoming or one outgoing edge)
    one_in = np.where((d_in == 1) & (d_out == 0))[0]
    one_out = np.where((d_in == 0) & (d_out == 1))[0]

    edges = []
    edges.append(np.column_stack((A[:, one_in].argmax(0).A1, one_in)))
    edges.append(np.column_stack((one_out, A[one_out].argmax(1).A1)))
    edges = np.row_stack(edges)

    edge_cover_set = set(map(tuple, edges))
    nodes = set(edges.flatten())

    # greedly add other edges such that both end-point are not yet in the edge_cover_set
    cands = np.column_stack(A.nonzero())
    for u, v in cands[d_in[cands[:, 1]].argsort()]:
        if u not in nodes and v not in nodes and u != v:
            edge_cover_set.add((u, v))
            nodes.add(u)
            nodes.add(v)
        if len(nodes) == N:
            break

    # add a single edge for the rest of the nodes not covered so far
    not_covered = np.setdiff1d(np.arange(N), list(nodes))
    edges = [list(edge_cover_set)]
    not_covered_out = not_covered[d_out[not_covered] > 0]

    if len(not_covered_out) > 0:
        edges.append(np.column_stack((not_covered_out, A[not_covered_out].argmax(1).A1)))

    not_covered_in = not_covered[d_out[not_covered] == 0]
    if len(not_covered_in) > 0:
        edges.append(np.column_stack((A[:, not_covered_in].argmax(0).A1, not_covered_in)))

    edges = np.row_stack(edges)

    # make sure that we've indeed computed an edge_cover
    # assert A[edges[:, 0], edges[:, 1]].sum() == len(edges)
    assert len(set(map(tuple, edges))) == len(edges)
    assert len(np.unique(edges)) == N

    return edges


def edges_to_sparse(edges, N, values=None):
    if values is None:
        values = np.ones(edges.shape[0])

    return sp.coo_matrix((values, (edges[:, 0], edges[:, 1])), shape=(N, N)).tocsr()


def score_link_prediction(labels, scores):
    return roc_auc_score(labels, scores), average_precision_score(labels, scores)