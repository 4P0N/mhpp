#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import argparse
from model import GLACE
from utils import DataUtils, score_link_prediction
import pickle
import time
import scipy.sparse as sp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit, train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import normalize
from sklearn.linear_model import LinearRegression
#from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import random
import copy
from IPython.display import clear_output
import random

node1_start, node1_end = (0, 218)
node2_start, node2_end = (218, 13557)
node3_start, node3_end = (13557, 14266)
node4_start, node4_end = (14266, 67117)

seed = 6521
MODEL_ID = 11
tensorboard_path = '../../1.Dataset/Processed_Dataset/House Price Content List/Tensorboard/'
model_path = '../../1.Dataset/Processed_Dataset/House Price Content List/Model Params/' 
embedding_path = '../../1.Dataset/Processed_Dataset/House Price Content List/Embeddings/'

def train_test_split_sampled(arrName, pivotColumn, num_of_intervals):
    arrName = arrName[~np.isnan(arrName).any(axis=1)]
    col = np.array(arrName[pivotColumn])
    #print(col)
    bins = np.linspace(min(col), max(col), num_of_intervals)
    left = 0
    right = 1
    train = np.array([arrName.values[0]])
    test_set = np.array([arrName.values[0]])
    train = np.delete(train,(0), axis = 0)
    #np.delete(val,(0), axis = 0)
    test_set = np.delete(test_set,(0), axis = 0)
    
    portion_arr = []
    while(right < len(bins)):
      left_val = bins[left]
      right_val = bins[right]
      portion_arr = arrName[arrName[pivotColumn] >= left_val]
      portion_arr = portion_arr[portion_arr[pivotColumn] < right_val]
      if len(portion_arr) < 10 and right !=len(bins) - 1:
        right = right + 1
        continue
      train_temp, test_temp = train_test_split(portion_arr, test_size = 0.2, random_state = seed)
      #print(train_temp.values.shape)
      #print(train.shape)
      train = np.concatenate((train, np.array(train_temp.values)))
      #val_temp, test_temp = train_test_split(val_test_temp, test_size = 0.5)
      test_set = np.concatenate((test_set, np.array(test_temp.values)))

      left = right
      right = right + 1
      portion_arr = []

    return train, test_set

def score_node_classification(features, z, features_test, labels_test, p_labeled=0.8, n_repeat=10, norm=False):
    """
    Train a classifier using the node embeddings as features and reports the performance.
    Parameters
    ----------
    features : array-like, shape [N, L]
        The features used to train the classifier, i.e. the node embeddings
    z : array-like, shape [N]
        The ground truth labels
    p_labeled : float
        Percentage of nodes to use for training the classifier
    n_repeat : int
        Number of times to repeat the experiment
    norm
    Returns
    -------
    f1_micro: float
        F_1 Score (micro) averaged of n_repeat trials.
    f1_micro : float
        F_1 Score (macro) averaged of n_repeat trials.
    """
    if p_labeled == 0.8:
        p_labeled = 1 - random.uniform(0.2, 0.8)
    if norm:
        features = normalize(features)

    trace = []
    split_train1, split_train2 = None, None
    for seed in range(n_repeat):
        sss = ShuffleSplit(n_splits=1, test_size=1 - p_labeled, random_state=seed)
        split_train, split_test = next(sss.split(features, z))

        rfr = RandomForestRegressor(n_jobs=-1)
        rfr.fit(features[split_train], z[split_train])
        predicted = rfr.predict(features[split_test])

        mae = mean_absolute_error(z[split_test], predicted)
        mse = mean_squared_error(z[split_test], predicted)
        
        rfr = RandomForestRegressor(n_jobs=-1)
        rfr.fit(features, z)
        predicted = rfr.predict(features_test)

        mae2 = mean_absolute_error(labels_test, predicted)
        mse2 = mean_squared_error(labels_test, predicted)


        trace.append((mae, mse, mae2, mse2))

    return np.array(trace).mean(0)


def check_performance(number_of_iter, test_indices):
    ps = pd.read_pickle(embedding_path + 'glace_cora_ml_embedding_graduate_second-order.pkl')
    #pf = pd.read_csv('features_concatenated.csv', sep=',', header=None)
    #raw_features = pf.values[14268:, 1:]
    features_train = np.array([np.array(ps['mu'][k]) for k in range(14266, len(ps['mu'])) if k not in test_indices])
    features_test = np.array([np.array(ps['mu'][k]) for k in range(14266, len(ps['mu'])) if k in test_indices])
    labels_df = pd.read_csv('Property_price.csv')
    labels_train = labels_df[~np.isin(labels_df['ID'], test_indices)]['price'].values
    labels_test = labels_df[np.isin(labels_df['ID'], test_indices)]['price'].values
    ''' for k in range(len(features)):
      if np.random.rand() < 0.00:
        print(str(k) + " feats: " +str(features[k]))'''

    mae, mse, mae2, mse2 = score_node_classification(features_train, labels_train, features_test, labels_test, n_repeat = 1) 
    if number_of_iter % 300 == 0 or number_of_iter < 1000:
      print("Embedding Features Results - MAE: "+ str(mae)+" RMSE: "+ str(mse**0.5)+" TMAE: "+ str(mae2)+" TRMSE: "+ str(mse2**0.5))
    return mse**0.5

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', default='cora_ml')
    parser.add_argument('model', default='glace', help='lace or glace')
    parser.add_argument('--suf', default='')
    parser.add_argument('--proximity', default='second-order', help='first-order or second-order')
    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--learning_rate', default=0.0002)
    parser.add_argument('--num_batches', type=int, default=600000)
    parser.add_argument('--is_all', default=True)  # train with all edges; no validation or test set
    args = parser.parse_args()
    args.is_all = True if args.is_all == 'True' else False
    train(args)

def train(args):
    """ graph_file = 'Data Tables Step 7 - Processed npzs_cleaned/graph_0_general_attributes.npz'
    graph_file1 = 'Data Tables Step 7 - Processed npzs_cleaned/graph_1_house_region.npz'
    graph_file2 = 'Data Tables Step 7 - Processed npzs_cleaned/graph_2_house_train.npz'
    graph_file3 = 'Data Tables Step 7 - Processed npzs_cleaned/graph_3_property_school.npz'
    graph_file4 = 'Data Tables Step 7 - Processed npzs_cleaned/graph_4_school_train.npz'
    graph_file5 = 'Data Tables Step 7 - Processed npzs_cleaned/graph_5_train_train.npz' """

    graph_file  = '../../1.Dataset/Processed_Dataset/graph_0_general_attributes.npz'
    graph_file1 = '../../1.Dataset/Processed_Dataset/graph_1_house_region.npz'
    graph_file2 = '../../1.Dataset/Processed_Dataset/graph_2_house_train.npz'
    graph_file3 = '../../1.Dataset/Processed_Dataset/graph_3_property_school.npz'
    graph_file4 = '../../1.Dataset/Processed_Dataset/graph_4_school_train.npz'
    graph_file5 = '../../1.Dataset/Processed_Dataset/graph_5_train_train.npz' 

    #Dataset splitting for ensuring inductivity 
    price_file = '../../1.Dataset/Processed_Dataset/Property_price.csv'
    df_price = pd.read_csv(price_file)
    train, tesst = train_test_split_sampled(df_price, 'price', 20)
    train_indices = train[:, 0]
    test_indices = tesst[:, 0]
    np.savetxt(embedding_path + "train.txt", train_indices)
    np.savetxt(embedding_path + "test.txt", test_indices)

    #Normal attribute and graph loaders, added with test indices for ensuring they don't get trained
    data_loader = DataUtils(graph_file, args.is_all, test_indices=test_indices) #THIS ONLY CONTAINS ATTRIBUTE INFO
    data_loader1 = DataUtils(graph_file1, args.is_all, data_loader.node_negative_distribution_temp,test_indices=test_indices)
    data_loader2 = DataUtils(graph_file2, args.is_all, data_loader.node_negative_distribution_temp,test_indices=test_indices)
    data_loader3 = DataUtils(graph_file3, args.is_all, data_loader.node_negative_distribution_temp,test_indices=test_indices)
    data_loader4 = DataUtils(graph_file4, args.is_all, data_loader.node_negative_distribution_temp,test_indices=test_indices)
    data_loader5 = DataUtils(graph_file5, args.is_all, data_loader.node_negative_distribution_temp,test_indices=test_indices)

    suffix = args.proximity
    args.X1 = data_loader.X1 if args.suf != 'oh' else sp.identity(data_loader1.X1.shape[0])
    args.X2 = data_loader.X2 if args.suf != 'oh' else sp.identity(data_loader2.X2.shape[0])
    args.X3 = data_loader.X3 if args.suf != 'oh' else sp.identity(data_loader3.X3.shape[0])
    args.X4 = data_loader.X4 if args.suf != 'oh' else sp.identity(data_loader4.X4.shape[0])

    m = args.model
    name = m + '_' + args.name
    if 'glace' == m:
        model = GLACE(args)
    else:
        raise Exception("Only glace available")

    writer = tf.summary.FileWriter(tensorboard_path)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        #saver.restore(sess, model_path+"model_graduate_best.ckpt")
        writer.add_graph(sess.graph)
        print('-------------------------- ' + m + ' --------------------------')
        print('batches\tloss\tsampling time\ttraining_time\tdatetime')

        tf.global_variables_initializer().run()
        sampling_time, training_time = 0, 0

        previous_best = 9999999999

        for b in range(args.num_batches):
            if b%35000 == 0:
              clear_output()
            t1 = time.time()
            #CREATE DIFFERENT SAMPLER
            if b%5 == 0:
                #sess.run(model.zero_ops)
                u_i, u_j, label, node_type1, node_type2 = data_loader1.fetch_next_batch(batch_size=args.batch_size, K=args.K)
            elif b%5 == 1:
                u_i, u_j, label, node_type1, node_type2 = data_loader2.fetch_next_batch(batch_size=args.batch_size, K=args.K)
            elif b%5 == 2:
                u_i, u_j, label, node_type1, node_type2 = data_loader3.fetch_next_batch(batch_size=args.batch_size, K=args.K)
            elif b%5 == 3:
                u_i, u_j, label, node_type1, node_type2 = data_loader4.fetch_next_batch(batch_size=args.batch_size, K=args.K)
            else:
                u_i, u_j, label, node_type1, node_type2 = data_loader5.fetch_next_batch(batch_size=args.batch_size, K=args.K)
            
            #u_i, u_j, label, w = data_loader.fetch_next_batch(batch_size=args.batch_size, K=args.K)
            feed_dict = {model.u_i: u_i, model.u_j: u_j, model.label: label, model.node_type1 : node_type1, model.node_type2 : node_type2}

            t2 = time.time()
            sampling_time += t2 - t1

            #loss, _ = sess.run([model.loss, model.accum_ops], feed_dict=feed_dict)
            loss, _ = sess.run([model.loss, model.train_op], feed_dict=feed_dict)
            ''' if (b%5 == 4 or True):
              sess.run(model.train_step) '''
            training_time += time.time() - t2

            if b%5 < 5:
              s = sess.run(model.merged_summary, feed_dict = feed_dict)
              writer.add_summary(s, b)
              writer.flush()


            if b % 5000 < 5:
                print('%d\t%f\t%0.2f\t%0.2f\t%s' % (b, loss, sampling_time, training_time,
                                                        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

                sampling_time, training_time = 0, 0 

            if b!=0 and ( b % 50 == 0 or b == (args.num_batches - 1)):
                if m == 'glace':
                    
                    mu1, sigma1 = sess.run([model.embedding1, model.sigma1])
                    mu2, sigma2 = sess.run([model.embedding2, model.sigma2])
                    mu3, sigma3 = sess.run([model.embedding3, model.sigma3])
                    mu4, sigma4 = sess.run([model.embedding4, model.sigma4])

                    mu = copy.deepcopy(mu1)
                    mu[node1_start: node1_end] = mu1[node1_start: node1_end]
                    mu[node2_start: node2_end] = mu2[node2_start: node2_end]
                    mu[node3_start: node3_end] = mu3[node3_start: node3_end]
                    mu[node4_start: node4_end] = mu4[node4_start: node4_end]

                    sigma = copy.deepcopy(sigma1)
                    sigma[node1_start: node1_end] = sigma1[node1_start: node1_end]
                    sigma[node2_start: node2_end] = sigma2[node2_start: node2_end]
                    sigma[node3_start: node3_end] = sigma3[node3_start: node3_end]
                    sigma[node4_start: node4_end] = sigma4[node4_start: node4_end]
                    
                    pickle.dump({'mu': data_loader.embedding_mapping(mu),
                                 'sigma': data_loader.embedding_mapping(sigma)},
                                open(embedding_path + '%s%s_embedding_graduate_%s.pkl' % (name, '_all' if args.is_all else '', suffix), 'wb'))
                    
                    save_path = saver.save(sess, model_path + "model_graduate.ckpt")
                    
                    curr_mae = check_performance(number_of_iter = b, test_indices = test_indices)

                    if curr_mae < previous_best:
                      previous_best = curr_mae
                      print("new best result: "+str(curr_mae))
                      pickle.dump({'mu': data_loader.embedding_mapping(mu),
                                 'sigma': data_loader.embedding_mapping(sigma)},
                                open(embedding_path + '%s%s_embedding_graduate_%s_best.pkl' % (name, '_all' if args.is_all else '', suffix), 'wb'))
                      save_path = saver.save(sess, model_path + "model_graduate_best.ckpt")

                else:
                    raise Exception("only GLACE supported")

if __name__ == '__main__':
    main()

                