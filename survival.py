# os-related imports
import sys
import csv
import time

# numpy
import numpy as np

# algorithms
from sklearn import ensemble, svm, tree, linear_model
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


def execute_survival(X, y, best_features, classifier):
    '''learn a given survival model'''

    y_for_cv = np.array([t[0] for t in y])
    cv = StratifiedKFold(y_for_cv, n_folds=10) # x-validation
    classifier = classifier()

    clf = Pipeline([('classifier',classifier)])

    CIscore = 0
    score_list = []
    
    print ('  ...performing x-validation')
    for i, (train, test) in enumerate(cv):
        print ('   ...',i+1)

        total = 0
        for target in y[train]:
            if target[0] == True:
                total += 1
                
        if total < 5:
            print ('...cannot train; too few positive examples')
            return False, False

        y_train = y[train]
        trained_classifier = clf.fit(X[train], y[train])
  
        event_indicators = []
        event_times = []
        scores = []

        # seperate the event indicators and the event times and store them in lists

        print('seperating event indicators and event times of fold',i+1)
        for target in y[test]:
           event_indicators.append(target[0])
           event_times.append(target[1])

        predictions = trained_classifier.predict(X[test])

        for prediction in predictions:
            scores.append(prediction)

        print('determining concordance index of fold',i+1)
        result = concordance_index_censored(np.array(event_indicators), np.array(event_times), np.array(scores).reshape(-1))
        print(result[0])
        CIscore += result[0]
        score_list.append(result[0])
        # TODO fix metrics

    avgCIscore = CIscore / len(cv)

    f = '/Users/Tristan/Downloads/scores.csv'
    support.iter_to_csv(score_list, f)
    low_bound, high_bound = support.confidence_interval(score_list)

    print ('  ...fitting model (full data sweep)'.format(X.shape))
    complete_classifier = clf.fit(X,y)

    print('fit complete')

    return avgCIscore, complete_classifier.named_steps['classifier']