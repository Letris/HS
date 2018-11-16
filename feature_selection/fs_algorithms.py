
import learn.models as ML
from sklearn.feature_selection import SelectKBest, f_classif, chi2, VarianceThreshold
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import simplejson
import pandas as pd
from tqdm import *
import util_.util as util
import util_.in_out as in_out
from sklearn import ensemble, svm, tree, linear_model
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso


def pearson_fs(X, y, k, headers, feature_selection, survival):
	k = k
	if feature_selection: #and X.shape[1] >= k
		print ('  ...performing pearson feature selection')
		
		if survival:
			pearson_y = []
			for tup in y:
				if tup[0] == False:
					pearson_y.append(0)
				else:
					pearson_y.append(1)
		else:
			pearson_y = y

		pearsons = []
		for i in range(X.shape[1]):
			p = pearsonr(np.squeeze(np.asarray(X[:,i])), pearson_y)
			pearsons.append(abs(p[0]))
		best_features = np.array(pearsons).argsort()[-k:][::-1]
		# print best_features

		new_headers = []
		test_list = best_features.tolist()

		# make sure none of the target values are present in the feature vector
		for i in best_features:
			if headers[i].upper()[0:3] in ['K90', 'K89', 'k90', 'k89', 'target', 'TARGET', 'tar', 'TAR']:
				print(headers[i])
				index = test_list.index(i)
				best_features = np.delete(best_features, index)
				test_list.pop(index)
				continue
			else:
				new_headers.append(headers[i])


		headers = new_headers
		new_X = X[:,best_features]
		
	else:
		new_X = X
		best_features='all'
		
	return new_X, best_features, headers


def lasso_fs(X, y, headers, k, feature_selection):
    print ('  ...performing lasso feature selection')
    if feature_selection and X.shape[1] >= k:
       
        lasso = Lasso()
        lasso.fit(X, y)
        scores = lasso.coef_

        index = 0
        features = dict()
        to_indeces = []

        for header in headers:
            features[header] = scores[index]
            index +=1

        sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)
        top_features = [key[0] for key in sorted_features if key[0].upper()[0:3] not in ['K90', 'K89', 'k90', 'k89', 'target', 'TARGET', 'tar', 'TAR']][0:k]
        print(top_features)

        for feature in top_features:
            if feature in headers:
                to_indeces.append(headers.index(feature))
        
        headers = top_features
        new_X = X[:,to_indeces]

        f = open('/Users/Tristan/Downloads/merged/important_features/lasso.txt', 'w')
        simplejson.dump(top_features, f)
        f.close()

        return new_X, top_features


def random_forest_fs(X, y, headers, k, feature_selection):

    if feature_selection and X.shape[1] >= k:
        print ('  ...performing random forest regressor feature selection')

        rf = RandomForestRegressor()
        rf.fit(X, y)

        print ("Features sorted by their score:")
        print (sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), headers), reverse=True))

        best_features = rf.feature_importances_.argsort()[::-1][:k]

        new_headers = []
        test_list = best_features.tolist()

        for i in best_features:
            if headers[i].upper()[0:3] in ['K90', 'K89', 'k90', 'k89', 'target', 'TARGET', 'tar', 'TAR']:
                print(headers[i])
                index = test_list.index(i)
                best_features = np.delete(best_features, index)
                test_list.pop(index)
                continue
            else:
                new_headers.append(headers[i])

        # headers = [headers[i] for i in best_features]

        headers = new_headers
        new_X = X[:,best_features]

    else:
        new_X = X
        best_features ='all'

    f = open('/Users/Tristan/Downloads/merged/important_features/RF.txt', 'w')
    simplejson.dump(headers, f)
    f.close()
    print(headers)
    return new_X, best_features


def RFE_fs(X, y, headers, k, feature_selection):

    if feature_selection and X.shape[1] >= k:
        print ('  ...performing RFE feature selection')

        model = linear_model.LogisticRegression()
        rfe = RFE(model, k)
        rfe.fit(X, y)

        index = 0
        new_headers = []
        for header in headers:
            if rfe.support_[index] == True and header.upper()[0:3] not in ['K90', 'K89', 'k90', 'k89', 'target', 'TARGET', 'tar', 'TAR']:
                new_headers.append(header)
                index +=1
            else:
                index +=1

        headers = new_headers
        new_X = X[headers]
       
        f = open('/Users/Tristan/Downloads/merged/important_features/RFE.txt', 'w')
        simplejson.dump(headers, f)
        f.close()

    print(headers)
    return new_X, headers


def Kbest_fs(X, y, headers, k, feature_selection):
    if feature_selection and X.shape[1] >= k:
        print ('  ...performing Kbest feature selection')

    test = SelectKBest(score_func=chi2, k=k)
    fit = test.fit(X, y)
    new_X = fit.transform(X)

    index = 0
    features = dict()
    scores = fit.scores_

    for header in headers:
        features[header] = scores[index]
        index +=1

    sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)
    top_features = [key[0] for key in sorted_features if key[0].upper()[0:3] not in ['K90', 'K89', 'k90', 'k89', 'tar', 'TAR', 'target', 'TARGET']][0:k]

    f = open('/Users/Tristan/Downloads/merged/important_features/kbest.txt', 'w')
    simplejson.dump(top_features, f)
    f.close()

    return new_X, top_features

