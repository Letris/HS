''' This file is used to perform hyperparameter tuning. You can change which parameters 
    and their values to tune in the file hpt.py file '''

from util_.in_out import import_data, dict2text
import feature_selection.fs as fs
from hpt import RandomGridSearchRFC_Fixed

# -------------------------------------------------------------------------------------------------------- parameters

# fill in the directory to the file that you want to use 
f = '/Users/Tristan/Downloads/data/'

# specify the identifier of a patient
record_id = 'ID'

# specifiy whether you are tuning parameters for survival or non-survival models (True/False)
survival = False

# specify the amount of features that the model has to use for training
k=150

# specify the models for whichh you want to perform hyperparameter tuning 
# options: cart, svm, rf, xgboost, lr, cox, survSVM, gb
models = ['lr']

# specify the dir where you want to save the output
out_dir = '/Users/Tristan/Downloads/data/hpt/'

# specify the amount of folds
splits = 2

#-------------------------------------------------------------------------------------------------------- runcode

# import data
x, y, headers, index_list = import_data(f, record_id, survival) 

# feature selection
new_X, best_features = fs.pearson_fs(x, y, headers, k, feature_selection=True, survival=False)

m_dict = dict()

for m in models:
    print('performing hyperparameter tuning for {}'.format(m))
    best_params, model = RandomGridSearchRFC_Fixed(new_X, y, splits, m, survival)
    m_dict[m] = best_params
    print('the best parameters for {} are {}'.format(m, best_params))

# save output
dict2text(out_dir, m_dict)

print('Finished hyperparameter tuning')
