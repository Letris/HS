# os-related imports
import sys
import csv
import time

# numpy
import numpy as np

# algorithms
from sklearn import ensemble, svm, tree, linear_model
sys.path.insert(0, '/Users/Tristan/Downloads/HS')
import support as support
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


def execute_non_survival(X, y, best_features, classifier, oversampling, undersampling):
    cv = StratifiedKFold(y, n_folds=10) # x-validation
    classifier = classifier()
    
    clf = Pipeline([('classifier',classifier)])

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    cm=np.zeros((2,2))
    score_list = []

    # cross fold validation
    print ('  ...performing x-validation')
    for i, (train, test) in enumerate(cv):
        print ('   ...',i+1)
        if sum(y[train]) < 5:
            print ('...cannot train; too few positive examples')
            return False, False

        # oversampling with SMOTE
        if oversampling == True:
            print ('performing oversampling')
            trained_classifier = perform_oversampling(X, y, clf, train)
            
        if undersampling ==  True:
            print('performing oversampling')
            trained_classifier = perform_undersampling(X, y, clf, train)

        else:
           print('performing normal fit')
           trained_classifier = clf.fit(X[train], y[train])


        y_pred = trained_classifier.predict_proba(X[test])
         
        # make cutoff for confusion matrix
        y_pred_binary = (y_pred[:,1] > 0.01).astype(int)

        # derive ROC/AUC/confusion matrix
        fpr, tpr, thresholds = roc_curve(y[test], y_pred[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        cm = cm + confusion_matrix(y[test], y_pred_binary) 
        score_list.append(auc(fpr, tpr))

    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    mean_cm = cm/len(cv)

    f = '/Users/Tristan/Downloads/scores.csv'
    support.iter_to_csv(score_list, f)
    low_bound, high_bound = support.confidence_interval(score_list)

    # redo with all data to return the features of the final model
    print ('  ...fitting model (full data sweep)'.format(X.shape))
    complete_classifier = clf.fit(X,y)
    # imp = importances(clf, X, y) # permutation
    # plot_importances(imp)
    print(mean_fpr, mean_tpr, mean_auc, mean_cm)
    return [mean_fpr, mean_tpr, mean_auc, mean_cm], complete_classifier.named_steps['classifier']

def perform_oversampling(x,y, clf, train):
    """Increases the amount of instances in the positive class"""
    sm = SMOTE(random_state=12, ratio = 1.0)
    x_train, y_train = sm.fit_sample(x[train], y[train])
    trained_classifier = clf.fit(x_train, y_train)
    return trained_classifier

def perform_undersampling(x, y, clf, train):
    """Reduces the amount of instances in the negative class"""
    rus = RandomUnderSampler(random_state=0)
    x_train, y_train = rus.fit_sample(x[train], y[train])
    trained_classifier = clf.fit(x_train, y_train)
    return trained_classifier

