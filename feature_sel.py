# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 14:44:34 2018
Workdir = F:\jTKount\1226 
Filename = feature_sel.py
Describe: Some basic method to select the feature; 
Reference: Luo Bin; blog:http://www.cnblogs.com/hhh5460/p/5186226.html
@author: OrenLi1042420545
"""

import numpy as np
import pandas as pd
from sklearn import datasets
from minepy import MINE
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import r2_score
from collections import defaultdict
from stability_selection import randomized_lasso
from sklearn.feature_selection import RFE

def distcorr(X, Y):
    """ Compute the distance correlation function
    
    >>> a = [1,2,3,4,5]
    >>> b = np.array([1,2,9,4,4])
    >>> distcorr(a, b)
    0.762676242417
    """
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    
    dcov2_xy = (A * B).sum()/float(n * n)
    dcov2_xx = (A * A).sum()/float(n * n)
    dcov2_yy = (B * B).sum()/float(n * n)
    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor



if __name__ == '__main__':
#  ############################################################################
    # dataset select   house  iris  cancer
    data_name = 'house'
    # method_choise 2  3  4  5
    method_choise = 2
    # method_name2  2: 'pearson'  'MIC'  'Distance'  'Model_based'
    method_name2 = 'Model_based'
    # method_name3  3: 'linear'   'lasso'   'ridge'
    method_name3 = 'ridge'
    # method_name4  4: 'MDI'  'MDA'
    method_name4 = 'MDI'
    # method_name5  5: 'Stab_sel'  'RFE'
    method_name5 = 'RFE'
#  ############################################################################
    # load data
    re_feMol = pd.DataFrame()
    if data_name == 'house':
        LSdat = datasets.load_boston()
    elif data_name == 'iris':
        LSdat = datasets.load_iris()
    elif data_name == 'cancer':
        LSdat = datasets.load_breast_cancer()
#  ############################################################################
    # get into model   
    if method_choise == 2:
        # 2 Univariate feature selection
        for i in range(LSdat.data.shape[1]):
            if method_name2 == 'pearson':
                re_feMol.loc[i, 'FeatrueImportance'] = np.corrcoef(
                        LSdat.target, LSdat.data[:, i])[0, 1]
            elif method_name2 == 'MIC': # notice H0
                m = MINE()
                m.compute_score(LSdat.target, LSdat.data[:, i])
                re_feMol.loc[i, 'FeatrueImportance'] = m.mic()
            elif method_name2 == 'Distance':
                re_feMol.loc[i, 'FeatrueImportance'] = distcorr(
                        LSdat.target, LSdat.data[:, i])
            elif method_name2 == 'Model_based':
                rf = RandomForestRegressor(n_estimators=20, max_depth=4)
                re_feMol.loc[i, 'FeatrueImportance'] = np.mean(cross_val_score(
                        rf, LSdat.data[:, i:i+1], LSdat.target, scoring='r2',
                        cv = ShuffleSplit(n_splits=10, 
                                          test_size=0.1, random_state=0)))
            re_feMol.loc[i, 'ind'] = 'Feature_' + str(i)
#  ############################################################################
    elif method_choise == 3:
        # 3. linear model and regex
        scaler = StandardScaler()
        if method_name3 == 'linear':
            lr = LinearRegression()
            lr.fit(scaler.fit_transform(LSdat.data), LSdat.target)
            re_feMol['FeatrueImportance'] = lr.coef_
        elif method_name3 == 'lasso':
            lasso = Lasso(alpha=.3)
            lasso.fit(scaler.fit_transform(LSdat.data), LSdat.target)
            re_feMol['FeatrueImportance'] = lasso.coef_
        elif method_name3 == 'ridge':
            ridge = Ridge(alpha=10)
            ridge.fit(scaler.fit_transform(LSdat.data), LSdat.target)
            re_feMol['FeatrueImportance'] = ridge.coef_      
        re_feMol['ind'] = ['Feature_' + str(i) for i in range(LSdat.data.shape[1])]
#  ############################################################################
    elif method_choise == 4:
        # 4. Random Forest
        if method_name4 == 'MDI':
            rf = RandomForestRegressor()
            rf.fit(LSdat.data, LSdat.target)
            re_feMol['FeatrueImportance'] = rf.feature_importances_
        if method_name4 == 'MDA':
            rf = RandomForestRegressor()
            names = LSdat.feature_names
            scores = defaultdict(list)
            for train_idx, test_idx in ShuffleSplit(n_splits=10,
                                                    test_size=0.1, 
                                                    random_state=0).split(LSdat.data):
                rf.fit(LSdat.data[train_idx], LSdat.target[train_idx])
                acc = r2_score(LSdat.target[test_idx],
                               rf.predict(LSdat.data[test_idx]))
                for i in range(LSdat.data.shape[1]):
                    X_t = LSdat.data[test_idx, :].copy()
                    np.random.shuffle(X_t[:, i])
                    shuff_acc = r2_score(LSdat.target[test_idx], rf.predict(X_t))
                    scores[names[i]].append((acc-shuff_acc)/acc)
            re_feMol['FeatrueImportance'] = [np.mean(score) for k, score in scores.items()]
        re_feMol['ind'] = ['Feature_' + str(i) for i in range(LSdat.data.shape[1])]
#  ############################################################################
    elif method_choise == 5:
        if method_name5 == 'Stab_sel':
            rlasso = randomized_lasso.RandomizedLasso(alpha=0.025)
            rlasso.fit(LSdat.data, LSdat.target)
            re_feMol['FeatrueImportance'] = rlasso.coef_
        re_feMol['ind'] = ['Feature_' + str(i) for i in range(LSdat.data.shape[1])]
        if method_name5 == 'RFE':
            lr = LinearRegression()
            rfe = RFE(lr, n_features_to_select=1)
            rfe.fit(LSdat.data, LSdat.target)
            re_feMol['FeatrueImportance'] = LSdat.data.shape[1] - rfe.ranking_
#  ############################################################################
    # table format
    re_feMol = re_feMol.set_index('ind')
    re_feMol['sort_help'] = re_feMol['FeatrueImportance'].abs()
    re_feMol = re_feMol.sort_values(
            by = 'sort_help', ascending=False).drop('sort_help', axis=1)  