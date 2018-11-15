# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, make_scorer
from scipy.stats import pearsonr
import matplotlib.pylab as plt
import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import csv
import sys
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from imblearn.over_sampling import SMOTE
from scipy import interp
from sksurv.preprocessing import OneHotEncoder
from sksurv.metrics import concordance_index_censored
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sklearn import ensemble, svm, tree, linear_model
from sklearn.feature_selection import SelectKBest, f_classif, chi2, VarianceThreshold
import simplejson
from skopt import BayesSearchCV
from sklearn.cross_validation import StratifiedKFold
from tqdm import *
import pandas as pd
import util_.util, util_.in_out, util.support


def RandomGridSearchRFC_Fixed(X,Y,splits, model, survival):
    """
    This function looks for the best set o parameters for RFC method
    Input: 
        X: training set
        Y: labels of training set
        splits: cross validation splits, used to make sure the parameters are stable
    Output:
        clf.best_params_: dictionary with the parameters, to use: param_svm['kernel']
    """    
      

    start_svm = time.time()  
    
    if model == 'svm':
        clf = svm.SVC()

        tuned_parameters = {
        'C': ([0.01, 1, 10]),
         'kernel': (['rbf', 'linear']),
        # 'kernel': (['linear', 'rbf', 'sigmoid']),
        # 'degree': ([1,3,5,10]),
        # 'decision_function_shape' : (['ovo', 'ovr']),
        # 'cache_size': ([500,1000,1500,2000]),
        'shrinking': ([False, True]),
        # 'probability': ([False, True])
        }
    
    if model == 'cart':
        clf = tree.DecisionTreeClassifier()

        tuned_parameters = {
        'criterion': (['gini', 'entropy']),
        'max_depth': ([10,20]),
        'min_samples_split': ([2,3,5]),
        'min_samples_leaf': ([2,3,5]),
        }

    if model == 'rf':
        clf = ensemble.RandomForestClassifier()
 
        tuned_parameters = {
        'n_estimators': ([200,500,1000]),
        # 'max_features': (['auto', 'sqrt', 'log2',1,4,8]),                   # precomputed,'poly', 'sigmoid'
        'max_depth':    ([10,20]),
        # 'criterion':    (['gini', 'entropy']),
        'min_samples_split':  [2,3,5],
        'min_samples_leaf':   [2,3,5],
        }
        
    if model == 'xgboost':
        clf = XGBClassifier()

        tuned_parameters = {
        'booster': (['gbtree']),
        'max_depth':   ([5,10,20]),
        'reg_lambda': ([0,1]),
        'reg_alpha': ([0,1]),
        'subsample': ([0.5,1])
        }

    if model == 'lr':
        clf = linear_model.LogisticRegression()

        tuned_parameters = {
        'solver': (['liblinear', 'sag', 'saga'])
        }

    if model == 'cox':
       
        clf =  CoxnetSurvivalAnalysis()
        tuned_parameters = {
        'n_alphas': ([50,100,200]),
        'l1_ratio': ([0.1,0.5,1]),

        }

    if model == 'survSVM':
        clf = FastSurvivalSVM()
        
        tuned_parameters = {
        'alpha': ([0.5,1]),
        'rank_ratio': ([0.5,1]),
        'max_iter': ([20,40,80]),
        'optimizer': (['rbtree', 'avltree']),
        }

    if model == 'gb':
        clf = GradientBoostingSurvivalAnalysis()
       
        tuned_parameters = {
        'learning_rate': ([0.1, 0.3]),
        'n_estimators': ([100,200,400]),
        'max_depth': ([3,6,12])        
        }

    
    if survival == True:
        scorer = make_scorer(CI, greater_is_better=True)

        y_for_cv = np.array([t[0] for t in Y])
        cv = StratifiedKFold(y_for_cv, n_folds=2) # x-validation

    else:
        cv = StratifiedKFold(Y, n_folds=2) # x-validation
        scores = ['roc_auc']   

    print ('  ...performing x-validation')
   
    clf =  GridSearchCV(clf, tuned_parameters, scoring='%s' % scores[0], cv=cv, verbose=10) #scoring='%s' % scores[0]
    # clf = BayesSearchCV(clf, tuned_parameters, n_iter=50, cv=splits,
    #                 optimizer_kwargs=dict(acq_func='LCB', base_estimator='RF'))

    clf.fit(X, Y)

    end_svm = time.time()
    print("Total time to process: ",end_svm - start_svm)
  
    return(clf.best_params_,clf)

sm = SMOTE(random_state=12, ratio = 1.0)

f = '/Users/Tristan/Downloads/FINAL_MODELS/merged_data/nonsurvFINAL/nonsurvFINAL.csv'#sys.argv[1]
target_id = 'target'
record_id = 'x'
survival = False
aggregation = True
k=150

x, y, headers, index_list = import_data(f, record_id, target_id, survival) # assumption: first column is patientnumber and is pruned, last is target

if aggregation == True:
    X, headers = aggregations(f, index_list, survival)

new_X, best_features = pearson_fs(X, y, headers, k, feature_selection=True, survival=False)

m_dict = dict()

for m in ['cart', 'svm', 'rf', 'xgboost', 'lr']: #'cart', 'svm', 'rf', 'xgboost', 'lr'
    best_params, model = RandomGridSearchRFC_Fixed(new_X, y, 2, m, survival)
    m_dict[m] = best_params
    print(m)
    print(best_params)
    print(m_dict)

print(m_dict)
