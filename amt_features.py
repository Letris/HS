# from util_.in_out import import_data
from execute_fs import execute_nonsurvival, execute_survival
import matplotlib.pyplot as plt
import csv 
import numpy as np
import pandas as pd


def amt_features(f, clf_binary, clf_survival, k_range, selector, survival):
    record_id = 'ID'
    target_id = 'target'

    X, y, headers, target_list = import_data(f, record_id, target_id, survival) # assumption: first column is patientnumber and is pruned, last is target
    print ('  ...instances: {}, attributes: {}'.format(X.shape[0], X.shape[1]))


    # lr = linear_model.LogisticRegression()

    auc_k_dict_pearson = dict()
    best_k = 0
    best_AUC = 0
    best_IC = 0


    for k in k_range: #range(25,1000,25)
        # y_for_cv = np.array([t[0] for t in y])
        if survival == True:
            mean_IC = execute_survival(X, y, k, headers, clf_survival, selector)
            auc_k_dict_pearson[k] = mean_IC
            if mean_IC > best_IC:
                best_k = k
        else:
            mean_AUC = execute_nonsurvival(X, y, k, headers, clf_binary, selector)
            auc_k_dict_pearson[k] = mean_AUC
            if mean_AUC > best_AUC:
                best_k = k
            
           
    lists_pearson = sorted(auc_k_dict_pearson.items()) # sorted by key, return a list of tuples
    
    x_pearson, y_pearson = zip(*lists_pearson) # unpack a list of pairs into two tuples
    
    print('best k is: {}'.format(best_k))

    return x_pearson, y_pearson

def plot_Kcurve(x, y, x1, y1, title, xlabel, ylabel, plot_file):
    ''' Function used to plot the k-AUC/IC trade-off curve''' 
    plt.plot(x, y, label='BE')
    plt.plot(x1, y1, label='PE')
    plt.ylim(0, 1)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()
    plt.savefig(plot_file)
    plt.show()

def import_data(f, record_id, target_id, survival):
	# '''imports the data and converts it to X (input) and y (output) data vectors'''

    rows = read_csv2(f)
    headers = get_headers(next(rows))

	# save and split records
    print ('  ...(loading)')
    records = [row[0:] for row in rows]
    print ('  ...(converting to matrix)')
    records = np.matrix(records)
    X = records[:,0:-1] # features
    headers = headers[0:-1]
        
	# output
    y = records[:,-1] # target

    if survival == False:
        y=np.squeeze(np.asarray(y.astype(np.float64)))

        print ('  ...(converting data type)')

        X = X.astype(np.float64, copy=False)
        y = y.astype(np.float64, copy=False)
        index_list = None
	
    if survival == True:
        target_list = []

        y=np.squeeze(np.asarray(y.astype(list)))
        X = X.astype(np.float64, copy=False)

        index_list = []
        for idx, target in tqdm(enumerate(y)):
            target = eval(target)
            tuple_target = tuple(target)
            if tuple_target[1] <= 0:
                print('yes sir')
                index_list.append(idx)
                continue

            target_list.append(tuple_target)

        
		# print(target_list)
        y = np.array(target_list, dtype=[('Status', '?'), ('Survival in days', '<f8')])

        X = np.delete(X, (index_list), axis=0)

	
        print ('  ...(converting data type)')

    return X, y, headers, index_list

def read_csv2(f, delim='\t'): #returns reader object which will iterate over lines in the given csv file
	'''opens a csv reader object'''
	return csv.reader(open(f, "r"), delimiter=delim) #was open(f,'rb')

def read_csv(f, delim=',', index_col='ID'):
	'''opens a csv reader object'''
	return pd.read_csv(f, sep=',', index_col=index_col, encoding = "ISO-8859-1") #index_col='pseudopatnummer'

def get_headers(row): #deze zelf toegevoegd uit util.py. want import util gaf error
	'''returns the non-capitalised and bugfixed version of the header'''
	headers = [el.lower() for el in row]
	headers[0] = headers[0].split("\xef\xbb\xbf")[1] if headers[0].startswith('\xef') else headers[0] # fix funny encoding problem
	return headers