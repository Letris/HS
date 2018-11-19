from amt_features import amt_features, plot_Kcurve
import matplotlib as plt
from sklearn import ensemble, svm, tree, linear_model
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis

#-------------------------------------------------------------------------- Parameters

# Specify the file that you want to use
f = ''

# specify where you want to save the final plot
plot_file = ''

# Specify the range of amount of features that you want to test 
k_range = range(25, 1000, 25) # (start, end, step)

# Specify whether the amount of features is determined for survival or non-survival (True/False)
survival = False

# Specify the base model for which the optimal amount of features is determined
# Example non-survival : linear_model.LogisticRegression()
# Example survival : CoxnetSurvivalAnalysis(l1_ratio=0.1, n_alphas=200)

clf = linear_model.LogisticRegression()

# Specify labels and title for plot
title = 'k_AUC trade-off curve'
ylabel = 'AUC'
xlabel = 'Number of features'

# --------------------------------------------------------------------------- execution code 

x, y = amt_features(f, survival, clf, k_range)

plot_Kcurve(x, y, title, xlabel, ylabel, plot_file)

