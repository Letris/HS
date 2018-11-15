
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

def Kbest_fs(X, y, headers, feature_selection, survival):
	k = 70

	if feature_selection and X.shape[1] >= k:
		print ('  ...performing RFE feature selection')
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
		top_features = [key[0] for key in sorted_features if key[0].upper()[0:3] not in ['K90', 'K89', 'k90', 'k89']][0:k]

		f = open('/Users/Tristan/Downloads/merged/important_features/kbest.txt', 'w')
		simplejson.dump(top_features, f)
		f.close()

		new_headers = [header for header in headers if header in top_features]

	else:
		new_X = X
		new_headers = headers
		top_features = 'all'

	return new_X, top_features, new_headers

def random_forest_fs(X, y, headers, feature_selection):

	k=50
	if feature_selection and X.shape[1] >= k:
		print ('  ...performing random forest regressor feature selection')

		rf = RandomForestRegressor()
		rf.fit(X, y)

		print ("Features sorted by their score:")
		print (sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), headers), reverse=True))

		best_features = rf.feature_importances_.argsort()[::-1][:k]
		headers = [headers[i] for i in best_features]
		new_X = X[:,best_features]

	else:
		new_X = X
		best_features ='all'
