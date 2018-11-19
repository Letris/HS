from util_.in_out import import_data
from execture_fs import execute_nonsurvival, execute_survival
import matplotlib as plt
def amt_features(f, survival, clf, k_range):
    record_id = 'ID'
    target_id = 'target'

    X, y, headers, target_list = import_data(f, record_id, survival) # assumption: first column is patientnumber and is pruned, last is target
    print ('  ...instances: {}, attributes: {}'.format(X.shape[0], X.shape[1]))


    # lr = linear_model.LogisticRegression()

    auc_k_dict_pearson = dict()
    best_k = 0
    best_AUC = 0
    best_IC = 0

    for k in k_range: #range(25,1000,25)
        # y_for_cv = np.array([t[0] for t in y])
        if survival == True:
            mean_IC = execute_survival(X, y, k, headers, clf)
            auc_k_dict_pearson[k] = mean_IC
            if mean_IC > best_IC:
                best_k = k
        else:
            mean_AUC = execute_nonsurvival(X, y, k, headers, clf)
            auc_k_dict_pearson[k] = mean_AUC
            if mean_AUC > best_AUC:
                best_k = k
            
           
    lists_pearson = sorted(auc_k_dict_pearson.items()) # sorted by key, return a list of tuples
    
    x_pearson, y_pearson = zip(*lists_pearson) # unpack a list of pairs into two tuples
    
    print('best k is: {}'.format(best_k))

    return x_pearson, y_pearson

def plot_Kcurve(x, y, title, xlabel, ylabel, plot_file):
    ''' Function used to plot the k-AUC/IC trade-off curve''' 
    plt.plot(x, y)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()
    plt.savefig(plot_file)
    plt.show()
