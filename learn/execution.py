
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
import feature_selection.fs as feature_selection

def execute(in_dir, out_dir, record_id, algorithms, feature_selection, survival, oversampling, undersampling):
	'''executes the learning task on the data in in_dir with the algorithms in algorithms.
		The results are written to out_dir and subdirectories,
	    and the record_ and target_ids are used to differentiate attributes and non-attributes'''
	print ('### executing learning algorithms on... ###')
	
	# get the files
	files = util.list_dir_csv(in_dir)

	# stop if no files found
	if not files:
		print ('No appropriate csv files found. Select an input directory with appropriate files')
		return

	files_test = files

	# create directory
	util.make_dir(out_dir)

	# execute each algorithm
	for alg in algorithms:
		print ('...{}'.format(alg))
	
		util.make_dir(out_dir+'/'+alg+'/')
		results_list = []	

		# list which will contain the results
	
		# run algorithm alg for each file f
		for f, f_test in zip(files,files_test):
			fname = in_out.get_file_name(f, extension=False)
			print (' ...{}'.format(fname))
	
			# get data, split in features/target. If invalid stuff happened --> exit
			X, y, headers, target_list = in_out.import_data(f, record_id, survival) # assumption: first column is patientnumber and is pruned, last is target
			if type(X) == bool: return
	

			print ('  ...instances: {}, attributes: {}'.format(X.shape[0], X.shape[1]))

			# train model and return model and best features
			model, best_features, results = execute_with_algorithm(alg, X, y, fname, headers, out_dir+'/'+alg+'/', record_id, feature_selection, oversampling, survival, undersampling)
			results_list.append(results)

		try:
			in_out.save_ROC(out_dir+'/'+alg+'/'+"roc.png", results_list, title='ROC curve')
		except IndexError:
			pass
		
		try:
			in_out.save_ROC(out_dir+'/'+alg+'_test/'+"roc.png", results_list2, title='ROC curve')
		except NameError:
			pass

	# notify user
	print ('## Learning Finished ##')

def execute_with_algorithm(alg, X, y, fname, headers, out_dir, record_id, feature_selection, oversampling, survival, undersampling):
	'''execute learning task using the specified algorithm'''

	# feature selection
	# if survival == True and aggregation == True:
	# 	k=150
	# if survival == True and aggregation == False:
	# 	k=220
	# if survival == False and aggregation == True:
	# 	k=150
	# if survival == False and aggregation == False:
	# 	k=220

	k=220

	# perform feature selection
	new_X, best_features, headers = feature_selection.pearson_fs(X, y, k, headers, feature_selection, survival)

	# execute algorithm
	if alg == 'DT':
		results, model = ML.CART(new_X, y, best_features, out_dir+"{}.dot".format(fname), headers, oversampling, undersampling)  #out_dir+"{}.dot".format(fname)
	elif alg == 'RF':
		results, features, model = ML.RF(new_X, y, best_features,oversampling, undersampling, n_estimators=200)
	elif alg == 'RFsmall':
		results, features, model = ML.RF(new_X, y, best_features, oversampling, undersampling, n_estimators=100)
	elif alg == 'SVM':
		results, model = ML.SVM(new_X, y, best_features, oversampling, undersampling)
	elif alg == 'LR':
		results, features, model = ML.LR(new_X, y, best_features,oversampling, undersampling)
	elif alg == 'XGBoost':
		results, features, model = ML.XGBoost(new_X, y, best_features,oversampling, undersampling)
	if alg == 'COX':
		results, features, model = ML.COX(new_X, y, best_features, oversampling, undersampling)
	if alg == 'survSVM':
		results, features, model = ML.survSVM(new_X, y, best_features, oversampling, undersampling)
	if alg == 'GBS':
		results, features, model = ML.GradientBoostingSurvival(new_X, y, best_features, oversampling, undersampling)

	if not results:
		return


	if survival == False:
		in_out.save_results(out_dir+fname+'.csv', ["fpr", "tpr", "auc", "cm"], results, [sum(y),len(y)])
	# else:
		# in_out.save_results(out_dir+fname+'.csv', ["CI"], results, [sum(y),len(y)])

	if 'features' in locals():
		features = features.flatten()
		in_out.save_features(out_dir+"features_" + fname + '.csv', zip(headers[1:-1], features))
	
	return model, best_features, [fname] + results[0:3]


def read_csv(f, delim=',', index_col='ID'):
	'''opens a csv reader object'''
	# return csv.reader(open(f, 'r'), delimiter=delim)
	return pd.read_csv(f, sep=',', index_col=index_col, encoding = "ISO-8859-1") #index_col='pseudopatnummer'