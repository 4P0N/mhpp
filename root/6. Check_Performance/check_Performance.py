import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import normalize
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import random
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import plot_importance
import xgboost as xgb
import lightgbm as lgb
import seaborn as sns
import copy
from sklearn.svm import SVR
from sklearn.decomposition import PCA


LOG_NORMALIZATION = True


def mean_absolute_percentage_error(y_true, y_pred):

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# learning_rate=0.05, loss=ls, max_depth=3, max_features=auto, min_samples_leaf=15, min_samples_split=20, n_estimators=1000, warm_start=True


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # Do the predictions of all base models on the test data and use the averaged predictions as
    # meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X)
                            for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self

    # Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)


def score_node_classification(features, z, p_labeled=0.8, n_repeat=10, norm=False, model_type=0):
    z_original = copy.deepcopy(z)
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

    if norm:
        features = normalize(features)

    trace = []
    trace_clean = []

    trace1 = []
    trace2 = []
    trace3 = []
    trace4 = []
    trace5 = []
    trace6 = []

    for seed in range(n_repeat):
        sss = ShuffleSplit(n_splits=1, test_size=1 -
                           p_labeled, random_state=seed)
        split_train, split_test = next(sss.split(features, z))
        split_train = np.loadtxt('embedding_1/train.txt', dtype=int)-14266
        split_test = np.loadtxt('embedding_1/test.txt', dtype=int)-14266

        lasso = Lasso(alpha=0.0001, copy_X=True, fit_intercept=True, max_iter=1000,
                      normalize=False, positive=False, precompute=False, random_state=1552,
                      selection='cyclic', tol=0.0001, warm_start=False)
        ENet = ElasticNet(alpha=0.0001, copy_X=True, fit_intercept=True, l1_ratio=0.2,
                          max_iter=1000, normalize=False, positive=False, precompute=False,
                          random_state=39, selection='cyclic', tol=0.0001, warm_start=False)
        KRR = KernelRidge(alpha=0.05, coef0=0.5, degree=2, gamma=0.005, kernel='laplacian',
                          kernel_params=None)

        svr = SVR(C=1.2, cache_size=2000, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
                  kernel='rbf', max_iter=-1, shrinking=False, tol=0.0001, verbose=False)

        GBoost = GradientBoostingRegressor(alpha=0.9, ccp_alpha=0.0, criterion='friedman_mse',
                                           init=None, learning_rate=0.005, loss='huber',
                                           max_depth=10, max_features='sqrt',
                                           max_leaf_nodes=None, min_impurity_decrease=0.0,
                                           min_impurity_split=None, min_samples_leaf=15,
                                           min_samples_split=10, min_weight_fraction_leaf=0.0,
                                           n_estimators=3000, n_iter_no_change=None,
                                           random_state=358, subsample=0.8,
                                           tol=0.0001, validation_fraction=0.1, verbose=0,
                                           warm_start='True')

        model_xgb = xgb.XGBRegressor(alpha=0.6, base_score=0.5, booster=None, colsample_bylevel=1,
                                     colsample_bynode=1, colsample_bytree=0.2, gamma=5e-05, gpu_id=-1,
                                     importance_type='gain', interaction_constraints=None,
                                     learning_rate=0.01, max_delta_step=0, max_depth=5,
                                     min_child_weight=1.9, monotone_constraints=None,
                                     n_estimators=6000, n_jobs=-1, num_parallel_tree=1,
                                     objective='reg:squarederror', random_state=328,
                                     reg_alpha=0.600000024, reg_lambda=1, scale_pos_weight=1,
                                     subsample=0.8, tree_method=None, validate_parameters=False,
                                     verbosity=1)

        model_lgb = lgb. LGBMRegressor(bagging_fraction=0.8, bagging_freq=5, bagging_seed=9,
                                       boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
                                       feature_fraction=0.2319, feature_fraction_seed=9, importance_type='split', learning_rate=0.005, max_depth=-1,
                                       min_child_samples=20, min_child_weight=0.001, min_data_in_leaf=15,
                                       min_split_gain=0.0, n_estimators=6000, n_jobs=-1, num_leaves=50,
                                       objective='fair', random_state=None, reg_alpha=0.0,
                                       reg_lambda=0.0, silent=True, subsample=0.8,
                                       subsample_for_bin=200000, subsample_freq=0)

        averaged_models = AveragingModels(models=(GBoost, KRR, model_xgb))

        stacked_averaged_models = StackingAveragedModels(base_models=(GBoost, model_xgb),
                                                         meta_model=KRR)

        model_list = [lasso, ENet, KRR, GBoost, model_xgb,
                      averaged_models, stacked_averaged_models, model_lgb, svr]

        if LOG_NORMALIZATION:

            model_list[model_type].fit(
                features[split_train], np.log1p(z[split_train]))
            predicted = np.expm1(
                model_list[model_type].predict(features[split_test]))

        else:
            averaged_models.fit(features[split_train], (z[split_train]))
            predicted = (averaged_models.predict(features[split_test]))

        for k in range(0, len(predicted)):
            if random.random() < 0.0:
                print(str(z[split_test][k]) +
                      "    ----    " + str(predicted[k]))

        """ sns.boxplot(x = z[split_test])"""
        # newly added metrics
        price_limits = np.log1p([180000, 450132, 610722, 831759, 2090052])

        mae = mean_absolute_error(z[split_test], predicted)
        mse = mean_squared_error(z[split_test], predicted)
        mape = mean_absolute_percentage_error(z[split_test], predicted)
        mape_exp = mean_absolute_percentage_error(
            np.exp(z[split_test]), np.exp(predicted))

        all_gt = z[split_test]

        ind = np.where((all_gt < price_limits[0]))
        gt1 = all_gt[ind]
        pr1 = predicted[ind]

        ind = np.where((all_gt >= price_limits[0]) & (
            all_gt < price_limits[1]))
        gt2 = all_gt[ind]
        pr2 = predicted[ind]

        ind = np.where((all_gt >= price_limits[1]) & (
            all_gt < price_limits[2]))
        gt3 = all_gt[ind]
        pr3 = predicted[ind]

        ind = np.where((all_gt >= price_limits[2]) & (
            all_gt < price_limits[3]))
        gt4 = all_gt[ind]
        pr4 = predicted[ind]

        ind = np.where((all_gt >= price_limits[3]) & (
            all_gt < price_limits[4]))
        gt5 = all_gt[ind]
        pr5 = predicted[ind]

        ind = np.where((all_gt >= price_limits[4]))
        gt6 = all_gt[ind]
        pr6 = predicted[ind]

        ind = np.where((all_gt >= price_limits[0]) & (
            all_gt < price_limits[4]))
        gt_clean = all_gt[ind]
        pr_clean = predicted[ind]

        mae1 = mean_absolute_error(gt1, pr1)
        mse1 = mean_squared_error(gt1, pr1)

        mae2 = mean_absolute_error(gt2, pr2)
        mse2 = mean_squared_error(gt2, pr2)

        mae3 = mean_absolute_error(gt3, pr3)
        mse3 = mean_squared_error(gt3, pr3)

        mae4 = mean_absolute_error(gt4, pr4)
        mse4 = mean_squared_error(gt4, pr4)

        mae5 = mean_absolute_error(gt5, pr5)
        mse5 = mean_squared_error(gt5, pr5)

        mae6 = mean_absolute_error(gt6, pr6)
        mse6 = mean_squared_error(gt6, pr6)

        mae_clean = mean_absolute_error(gt_clean, pr_clean)
        mse_clean = mean_squared_error(gt_clean, pr_clean)

        trace.append((mae, mse, mape, mape_exp))
        trace_clean.append((mae_clean, mse_clean))
        trace1.append((mae1, mse1))
        trace2.append((mae2, mse2))
        trace3.append((mae3, mse3))
        trace4.append((mae4, mse4))
        trace5.append((mae5, mse5))
        trace6.append((mae6, mse6))

    return np.array(trace).mean(0), np.array(trace_clean).mean(0), np.array(trace1).mean(0), np.array(trace2).mean(0), np.array(trace3).mean(0), np.array(trace4).mean(0), np.array(trace5).mean(0), np.array(trace6).mean(0)


def check_performance(test_type="regular"):
    # Test type can be either "GSNE" or "regular"
    assert(test_type == "first" or test_type == "regular" or test_type ==
           "second" or test_type == "first+second", "Unknown test type.")

    # folder embedding_1 is first order, folder embedding_2 is second order
    ps = pd.read_pickle(
        'embedding_1/glace_cora_ml_embedding_graduate_second-order_best.pkl')
    ps1 = pd.read_pickle(
        'embedding_2/glace_cora_ml_embedding_graduate_second-order_best.pkl')

    pf = pd.read_csv(
        '../1.Dataset/Check_Performance_Dataset/property.csv').values[:, 1:]

    features = np.array([np.array(ps['mu'][k])
                        for k in range(14266, len(ps['mu']))])
    features1 = np.array([np.array(ps1['mu'][k])
                         for k in range(14266, len(ps1['mu']))])

    labels = pd.read_csv(
        '../1.Dataset/Check_Performance_Dataset/Property_price.csv').values[:, 1]

    if test_type == "regular":
        which = pf
    elif test_type == "first":
        which = np.hstack((pf, features))
    elif test_type == "second":
        which = np.hstack((pf, features1))
    else:
        which = np.hstack((pf, features, features1))

    repeats = 1

    print("lasso:")
    (mae, mse, mape, mape_exp), (mae_clean, mse_clean), (mae1, mse1), (mae2, mse2), (mae3, mse3), (mae4, mse4), (mae5,
                                                                                                                 mse5), (mae6, mse6) = score_node_classification(which, np.log1p(labels), n_repeat=repeats, model_type=0)
    print("======Embedding Features Results========")
    print("MAE: " + str(mae)+" RMSE: " + str(mse**0.5) + " MAPE: " + str(mape) + " MAPE_exp: " +
          str(mape_exp) + "MAE_cleaned: " + str(mae_clean)+" RMSE_cleaned: " + str(mse_clean**0.5))

    print("ENET:")
    (mae, mse, mape, mape_exp), (mae_clean, mse_clean), (mae1, mse1), (mae2, mse2), (mae3, mse3), (mae4, mse4), (mae5,
                                                                                                                 mse5), (mae6, mse6) = score_node_classification(which, np.log1p(labels), n_repeat=repeats, model_type=1)
    print("======Embedding Features Results========")
    print("MAE: " + str(mae)+" RMSE: " + str(mse**0.5) + " MAPE: " + str(mape) + " MAPE_exp: " +
          str(mape_exp) + "MAE_cleaned: " + str(mae_clean)+" RMSE_cleaned: " + str(mse_clean**0.5))

    print("KRR:")
    (mae, mse, mape, mape_exp), (mae_clean, mse_clean), (mae1, mse1), (mae2, mse2), (mae3, mse3), (mae4, mse4), (mae5,
                                                                                                                 mse5), (mae6, mse6) = score_node_classification(which, np.log1p(labels), n_repeat=repeats, model_type=2)
    print("======Embedding Features Results========")
    print("MAE: " + str(mae)+" RMSE: " + str(mse**0.5) + " MAPE: " + str(mape) + " MAPE_exp: " +
          str(mape_exp) + "MAE_cleaned: " + str(mae_clean)+" RMSE_cleaned: " + str(mse_clean**0.5))

    print("GBoost:")
    (mae, mse, mape, mape_exp), (mae_clean, mse_clean), (mae1, mse1), (mae2, mse2), (mae3, mse3), (mae4, mse4), (mae5,
                                                                                                                 mse5), (mae6, mse6) = score_node_classification(which, np.log1p(labels), n_repeat=repeats, model_type=3)
    print("======Embedding Features Results========")
    print("MAE: " + str(mae)+" RMSE: " + str(mse**0.5) + " MAPE: " + str(mape) + " MAPE_exp: " +
          str(mape_exp) + "MAE_cleaned: " + str(mae_clean)+" RMSE_cleaned: " + str(mse_clean**0.5))

    print("model_xgb:")
    (mae, mse, mape, mape_exp), (mae_clean, mse_clean), (mae1, mse1), (mae2, mse2), (mae3, mse3), (mae4, mse4), (mae5,
                                                                                                                 mse5), (mae6, mse6) = score_node_classification(which, np.log1p(labels), n_repeat=repeats, model_type=4)
    print("======Embedding Features Results========")
    print("MAE: " + str(mae)+" RMSE: " + str(mse**0.5) + " MAPE: " + str(mape) + " MAPE_exp: " +
          str(mape_exp) + "MAE_cleaned: " + str(mae_clean)+" RMSE_cleaned: " + str(mse_clean**0.5))

    print("averaged models:")
    (mae, mse, mape, mape_exp), (mae_clean, mse_clean), (mae1, mse1), (mae2, mse2), (mae3, mse3), (mae4, mse4), (mae5,
                                                                                                                 mse5), (mae6, mse6) = score_node_classification(which, np.log1p(labels), n_repeat=repeats, model_type=5)
    print("======Embedding Features Results========")
    print("MAE: " + str(mae)+" RMSE: " + str(mse**0.5) + " MAPE: " + str(mape) + " MAPE_exp: " +
          str(mape_exp) + "MAE_cleaned: " + str(mae_clean)+" RMSE_cleaned: " + str(mse_clean**0.5))

    print("stacked avg models:")
    (mae, mse, mape, mape_exp), (mae_clean, mse_clean), (mae1, mse1), (mae2, mse2), (mae3, mse3), (mae4, mse4), (mae5,
                                                                                                                 mse5), (mae6, mse6) = score_node_classification(which, np.log1p(labels), n_repeat=repeats, model_type=6)
    print("======Embedding Features Results========")
    print("MAE: " + str(mae)+" RMSE: " + str(mse**0.5) + " MAPE: " + str(mape) + " MAPE_exp: " +
          str(mape_exp) + "MAE_cleaned: " + str(mae_clean)+" RMSE_cleaned: " + str(mse_clean**0.5))

    print("LGB models:")
    (mae, mse, mape, mape_exp), (mae_clean, mse_clean), (mae1, mse1), (mae2, mse2), (mae3, mse3), (mae4, mse4), (mae5,
                                                                                                                 mse5), (mae6, mse6) = score_node_classification(which, np.log1p(labels), n_repeat=repeats, model_type=7)
    print("======Embedding Features Results========")
    print("MAE: " + str(mae)+" RMSE: " + str(mse**0.5) + " MAPE: " + str(mape) + " MAPE_exp: " +
          str(mape_exp) + "MAE_cleaned: " + str(mae_clean)+" RMSE_cleaned: " + str(mse_clean**0.5))


if __name__ == "__main__":
    check_performance("regular")
