from amt_features import amt_features, plot_Kcurve
import matplotlib as plt
from sklearn import ensemble, svm, tree, linear_model
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis

#-------------------------------------------------------------------------- Parameters

# Specify the file that you want to use
f_binary = '/Users/Tristan/Downloads/Preprocessed16clas.csv'
f_survival = '/Users/Tristan/Downloads/Preprocessed16classurv.csv'

# specify where you want to save the final plot
plot_file = '/Users/Tristan/Downloads/data/'

# Specify the range of amount of features that you want to test 
k_range = range(10, 200, 10) # (start, end, step)

# Specify whether the amount of features is determined for survival or non-survival (True/False)
survival = False

# Specify the base model for which the optimal amount of features is determined
# Example non-survival : linear_model.LogisticRegression()
# Example survival : CoxnetSurvivalAnalysis(l1_ratio=0.1, n_alphas=200)

clf_binary = ensemble.RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=5, min_samples_split=2, n_jobs=-1)
clf_survival = CoxnetSurvivalAnalysis(l1_ratio=0.1, n_alphas=200)

# Specify labels and title for plot
title = 'k_AUC trade-off curve'
ylabel = 'AUC'
xlabel = 'Number of features'

# --------------------------------------------------------------------------- execution code 

# define all experiments

# binary, pearson
x, y = amt_features(f_binary, clf_binary, clf_survival, k_range, selector='BE', survival=False)

# # binary, BE
x1, y1 = amt_features(f_binary, clf_binary, clf_survival, k_range, selector='PE', survival=False)

# # survival, pearson
# x, y = amt_features(f_survival, clf_binary, clf_survival, k_range, selector='PE', survival=True)

# # survival, BE
# x, y = amt_features(f_survival, clf_binary, clf_survival, k_range, selector='BE', survival=True)


plot_Kcurve(x, y, x1, y1, title, xlabel, ylabel, plot_file)

