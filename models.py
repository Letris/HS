# os-related imports
import sys
import csv
import time

# numpy
import numpy as np

# algorithms
from sklearn import ensemble, svm, tree, linear_model
from binary import execute_non_survival
from survival import execute_survival
# statistics, metrics, x-fold val, plots
from sklearn.metrics import roc_curve, auc, confusion_matrix, make_scorer
from sklearn.cross_validation import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.grid_search import RandomizedSearchCV
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from imblearn.over_sampling import SMOTE
from scipy import interp
from sksurv.preprocessing import OneHotEncoder
from sksurv.metrics import concordance_index_censored
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
import scipy as sp
import scipy.stats
from rfpimp import importances, plot_importances
from treeinterpreter import treeinterpreter as ti
import pandas as pd
from tqdm import *
from xgboost import plot_importance
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def SVM(X, y, best_features, oversampling, undersampling):
    
    clf = svm.SVC(probability=True, cache_size=1500, verbose=True, shrinking=False, C=0.001, kernel='linear') #probability=True, cache_size=1500, verbose=True, shrinking=False, C=0.001, kernel='linear'

    # e_clf = ensemble.BaggingClassifier(clf, n_estimators=1, max_samples = 0.2, n_jobs=-1, verbose=True)
    results, model = execute_non_survival(X, y, best_features, lambda: clf, oversampling, undersampling)
    return results, model

def CART(X, y, best_features, out_file, field_names, oversampling, undersampling):
   
    results, model = execute_non_survival(X, y, best_features, lambda: tree.DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_leaf=2, min_samples_split=2), oversampling, undersampling)
   
    if model:
        tree.export_graphviz(model, out_file=out_file, feature_names=field_names)
    return results, model

def RF(X, y, best_features, oversampling, undersampling, n_estimators):
    
    results, model = execute_non_survival(X, y, best_features, lambda: ensemble.RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=3, min_samples_split=5, n_jobs=-1), oversampling, undersampling)
   
    if model:
        features = model.feature_importances_
    else:
        features = False

    return results, features, model
   
def LR(X, y, best_features, oversampling, undersampling):
    
    results, model = execute_non_survival(X, y, best_features, lambda: linear_model.LogisticRegression(solver='liblinear', max_iter=10000), oversampling, undersampling)
    
    if model:
        features = model.coef_
    else:
        features = False

    return results, features, model

def XGBoost(X, y, best_features, oversampling, undersampling):
    
    results, model = execute_survival(X, y, best_features, lambda: XGBClassifier(booster='gbtree', max_depth=10, reg_alpha=1, reg_lambda=1, subsample=1))
    
    if model:
        features = model.feature_importances_
        plot_importance(model)
    else:
        features = False

    return results, features, model

# survival models
def COX(X, y, best_features, oversampling, undersampling):
    
    results, model = execute_survival(X, y, best_features,  lambda: CoxnetSurvivalAnalysis(l1_ratio=0.1, n_alphas=200))
    
    if model:
        features = model.coef_
    else:
        features = False

    return results, features, model

def GradientBoostingSurvival(X, y, best_features, oversampling, undersampling):
    results, model = execute_survival(X, y, best_features, lambda:GradientBoostingSurvivalAnalysis(learning_rate=0.1, max_depth=6, n_estimators=100))
    
    if model:
        features = model.feature_importances_
    else:
        features = False
    return results, features, model