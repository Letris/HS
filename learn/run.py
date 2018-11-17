'''this file is used to train models. Fill in the parameters and run the file.'''

from execution import execute


# fill in the directory where the input file (with feature vectors) is located
in_dir = "/Users/Tristan/Downloads/data/"

# fill in the directory where you want the output to be saved
out_dir = '/Users/Tristan/Downloads/data' #some dir

# Put the algorithms that you want to train in the following list
# Options: SVM, CART, RF LR, XGBoost, COX, survSVM, GBS
algorithms = ['LR', 'RF']

# Choose whether you want to use feature selection (True/False)
feature_selection = True
 
# Specify whether the algorithms you want to test are binary or survival (can not do both at the same time)
survival = False

# Specify the column that identies each unique patient 
record_id = 'ID'

# Specify whether you want to use oversampling (only for binary)
oversampling = False

# Specify whether you want to use undersampling (only for binary)
undersampling = False

execute(in_dir, out_dir, record_id, algorithms, feature_selection, survival, oversampling, undersampling)