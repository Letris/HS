''' This file is used to perform hyperparameter tuning. You can change which parameters 
    and their values to tune in the file hpt.py file '''
    
import sys
sys.path.insert(0, '/Users/Tristan/Downloads/HS/')
from in_out import import_data2, dict2text
import fs_algorithms as fs
from hpt import RandomGridSearchRFC_Fixed

# -------------------------------------------------------------------------------------------------------- parameters

# fill in the directory to the file that you want to use 
f = '/Users/Tristan/Downloads/data/nonsurvCRVMfinal.csv'

# specify the identifier of a patient
record_id = 'ID'
target_id = 'target'
# specifiy whether you are tuning parameters for survival or non-survival models (True/False)
survival = False

# specify the amount of features that the model has to use for training
k=150

# specify the models for whichh you want to perform hyperparameter tuning 
# options: cart, svm, rf, xgboost, lr, cox, survSVM, gb
models = ['lr']

# specify the dir where you want to save the output
out_dir = '/Users/Tristan/Downloads/data/hpt/hpt.txt'

# specify the amount of folds
splits = 2

#-------------------------------------------------------------------------------------------------------- runcode

# import data
x, y, headers, index_list = import_data2(f, record_id, target_id, survival) 

# feature selection
new_X, best_features, headers = fs.pearson_fs(x, y, headers, k, feature_selection=True, survival=survival)

m_dict = dict()

# perform hpt for each specified model
for m in models:
    print('performing hyperparameter tuning for {}'.format(m))
    best_params, model = RandomGridSearchRFC_Fixed(new_X, y, splits, m, survival)
    m_dict[m] = best_params
    print('the best parameters for {} are {}'.format(m, best_params))

# save output
dict2text(out_dir, m_dict)

print('Finished hyperparameter tuning')
