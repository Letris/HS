# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
from scipy import interp
from sklearn import ensemble, svm, tree, linear_model
from sklearn.ensemble import RandomForestRegressor
from tqdm import *
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sklearn import ensemble, svm, tree, linear_model
from fs_algorithms import pearson_fs


def execute_nonsurvival(X, y, k, headers, clf):
    
        new_X, best_features, headers = pearson_fs(X, y, headers, k, feature_selection=True, survival=False)
        cv = StratifiedKFold(y, n_folds=3) # x-validation 

        #clf = ensemble.RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=5, min_samples_split=2, n_jobs=-1)

        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        all_tpr = []
        cm=np.zeros((2,2))

        # cross fold validation
        print ('  ...performing x-validation')
        for i, (train, test) in enumerate(cv):
                print ('   ...',i+1)
                if sum(y[train]) < 5:
                    print ('...cannot train; too few positive examples')

                
                x_train, y_train = new_X[train], y[train]


                trained_classifier = clf.fit(x_train, y_train)
                
                y_pred = trained_classifier.predict_proba(new_X[test])
            
                # make cutoff for confusion matrix
                y_pred_binary = (y_pred[:,1] > 0.01).astype(int)

                # derive ROC/AUC/confusion matrix
                fpr, tpr, thresholds = roc_curve(y[test], y_pred[:, 1])
                mean_tpr += interp(mean_fpr, fpr, tpr)
                mean_tpr[0] = 0.0
                cm = cm + confusion_matrix(y[test], y_pred_binary) 

        mean_tpr /= len(cv)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        print(mean_auc)
        mean_cm = cm/len(cv)

        return mean_auc

def execute_survival(X, y, k, headers, clf):
    new_X, best_features, headers = pearson_fs(X, y, headers, k, feature_selection=True, survival=True)
    y_for_cv = np.array([t[0] for t in y])
    cv = StratifiedKFold(y_for_cv, n_folds=5) # x-validation
    
    #clf = CoxnetSurvivalAnalysis(l1_ratio=0.1, n_alphas=200)

    CIscore = 0
    
    print ('  ...performing x-validation')
    for i, (train, test) in enumerate(cv):
        print ('   ...',i+1)
    
        y_train = y[train]
        trained_classifier = clf.fit(new_X[train], y[train])


        event_indicators = []
        event_times = []
        scores = []

        for target in y[test]:
           event_indicators.append(target[0])
           event_times.append(target[1])

        predictions = trained_classifier.predict(new_X[test])

        for prediction in predictions:
            scores.append(prediction)

        result = concordance_index_censored(np.array(event_indicators), np.array(event_times), np.array(scores).reshape(-1))
        CIscore += result[0]
        # TODO fix metrics

    avgCIscore = CIscore / len(cv)
    print(avgCIscore)

    return avgCIscore