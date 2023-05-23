import numpy as np
from sklearn.metrics import accuracy_score # other metrics too pls!
from sklearn.ensemble import RandomForestClassifier # more!
# from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import itertools
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
print('\n')
print(f'Joaquin Dominguez')
print(f'ML2 - HW2')
print(f'---------------------------------------------------------------')
print(f'1. Write a function to take a list or dictionary of clfs and hypers(i.e. use logistic regression), each with 3 different sets of hyper parameters for each\n')

X,y = make_classification(n_samples = 1000, n_features= 15, n_repeated=0)

hypersLR = {
    'penalty' : ['l2', None],
    'C' : [1, 2, 5],
    'max_iter' : [100, 200, 500]
}
print(f'Hyperparameters: {hypersLR}\n')

def LR_GS(data_X, data_y, clf_hyper={}):
    #getting keys and values from param dict
    keys,values = zip(*clf_hyper.items())
    # creating dicts of all possible combinations of params
    permutations_dicts = [dict(zip(keys,v)) for v in itertools.product(*values)]
    # fitting a new model for every combination of dict
    best = 0
    for i in range(len(permutations_dicts)):
        clf = LogisticRegression(**permutations_dicts[i])
        clf.fit(data_X,data_y)
        # we could change to test data here if we wanted, but for the sake of this function, this works
        score = clf.score(data_X,data_y)
        if score > best:
            best = score
            best_params = clf.get_params()
    # print(f'Best Params: {best_params}')
    # print(f'Accuracy: {best}')
    # return best parameters
    ret = {}
    ret['best_params'] = best_params
    ret['accuracy_score'] = best
    return ret

print(f'Logistic Regression model with best accuracy and respective parameters:')
print(LR_GS(X, y, hypersLR))

print (f'-------------------------------------------------------')
print(f'2. Expand to include larger number of classifiers and hyperparameter settings\n')

hypersLR_RF = {'LR':{
    'penalty' : ['l2', None],
    'C' : [1, 2, 5],
    'max_iter' : [100, 200, 500]
    },
    'RF':{'n_estimators': [100,150,250],
    'criterion':['gini', 'entropy', 'log_loss'],
    'bootstrap': [True, False]}
}

clfs = [LogisticRegression(), RandomForestClassifier()]

print(f'Hyperparameters for LR and RF models: {hypersLR_RF}\n')

def LRRF_GS(data_X, data_y, clf_hyper={}, clf_list=[]):
    best = 0
    best_params = {}
    for i in clf_list:
        if type(i) == type(LogisticRegression()):
            LR_params = clf_hyper['LR']
            #getting keys and values from param dict
            keys,values = zip(*LR_params.items())
            # creating dicts of all possible combinations of params
            permutations_dicts = [dict(zip(keys,v)) for v in itertools.product(*values)]
            for i in range(len(permutations_dicts)):
                clf = LogisticRegression(**permutations_dicts[i])
                clf.fit(data_X,data_y)
                # we could change to test data here if we wanted, but for the sake of this function, this works
                score = clf.score(data_X,data_y)
                if score > best:
                    clf_type = 'LogisticRegression'
                    best = score
                    best_params = clf.get_params()
        else:
            RF_params = clf_hyper['RF']
            #getting keys and values from param dict
            keys,values = zip(*RF_params.items())
            # creating dicts of all possible combinations of params
            permutations_dicts = [dict(zip(keys,v)) for v in itertools.product(*values)]
            for i in range(len(permutations_dicts)):
                clf = RandomForestClassifier(**permutations_dicts[i])
                clf.fit(data_X,data_y)
                # we could change to test data here if we wanted, but for the sake of this function, this works
                score = clf.score(data_X,data_y)
                if score > best:
                    clf_type = 'RandomForest'
                    best = score
                    best_params = clf.get_params()
    # print(f'Best Params: {best_params}')
    # print(f'Accuracy: {best}')
    # return best parameters
    ret = {}
    ret['Type'] = clf_type
    ret['accuracy_score'] = best
    ret['best_params'] = best_params

    return ret

print(f'Best accuracy score and respective parameters from Logistic Regression and Random Forest grid search:')
print(LRRF_GS(X, y, hypersLR_RF, clfs))

print (f'-------------------------------------------------------')
print(f'3. Find some simple data \n Response: Simple data was generated using Numpy\'s make_classification() function \n4. Generate matplotlib plots that will assist in identifying the optimal clf and parampters settings\n')

def LRRF_GSopt(data_X, data_y, clf_hyper={}, clf_list=[]):
    best = 0
    best_params = {}
    counter_LR = 1
    counter_RF = 1
    viz_LR = {}
    viz_RF = {}
    for i in clf_list:
        if type(i) == type(LogisticRegression()):
            LR_params = clf_hyper['LR']
            #getting keys and values from param dict
            keys,values = zip(*LR_params.items())
            # creating dicts of all possible combinations of params
            permutations_dicts = [dict(zip(keys,v)) for v in itertools.product(*values)]
            for i in range(len(permutations_dicts)):
                clf = LogisticRegression(**permutations_dicts[i])
                clf.fit(data_X,data_y)
                # we could change to test data here if we wanted, but for the sake of this function, this works
                score = clf.score(data_X,data_y)
                label = str(counter_LR) + ' LR'
                viz_LR[label] = [score, clf.get_params()]
                counter_LR+=1
                if score > best:
                    clf_type = 'LogisticRegression'
                    best = score
                    best_params = clf.get_params()
        else:
            RF_params = clf_hyper['RF']
            #getting keys and values from param dict
            keys,values = zip(*RF_params.items())
            # creating dicts of all possible combinations of params
            permutations_dicts = [dict(zip(keys,v)) for v in itertools.product(*values)]
            for i in range(len(permutations_dicts)):
                clf = RandomForestClassifier(**permutations_dicts[i])
                clf.fit(data_X,data_y)
                # we could change to test data here if we wanted, but for the sake of this function, this works
                score = clf.score(data_X,data_y)
                label = str(counter_RF) + ' RF'
                viz_RF[label] = [score, clf.get_params()]
                counter_RF+=1
                if score > best:
                    clf_type = 'RandomForest'
                    best = score
                    best_params = clf.get_params()
    ret = {}
    ret['Type'] = clf_type
    ret['accuracy_score'] = best
    ret['best_params'] = best_params

    return ret, viz_LR, viz_RF 

ret, viz_LR, viz_RF = LRRF_GSopt(X, y, hypersLR_RF, clfs)

def viz_param(model, parameter):
    if model == 'LR':
        if parameter in viz_LR['1 LR'][1].keys():
            val_list = []
            param_list = []
            for i in range(1, len(viz_LR)+1):
                label = str(i) + ' LR'
                val_acc = viz_LR[label][0]
                val_list.append(val_acc)
                param_val = viz_LR[label][1][parameter]
                param_list.append(param_val)
            plt.rcParams["figure.figsize"] = (20,3)
            plt.title(f'Logistic Regression Accuracy by Parameter: \'{parameter}\'')
            plt.scatter(param_list, val_list, label = parameter, color = 'red')
            plt.show()
    else:
        if parameter in viz_RF['1 RF'][1].keys():
            val_list = []
            param_list = []
            for i in range(1, len(viz_RF)+1):
                label = str(i) + ' RF'
                val_acc = viz_RF[label][0]
                val_list.append(val_acc)
                param_val = viz_RF[label][1][parameter]
                param_list.append(param_val)
            plt.rcParams["figure.figsize"] = (20,3)
            plt.title(f'Random Forest Accuracy by Parameter: \'{parameter}\'')
            plt.scatter(param_list, val_list, label = parameter, color = 'blue')
            plt.show()

viz_param('LR', 'C')

viz_param('RF', 'n_estimators')
print(f'\n')
print (f'-------------------------------------------------------')
print(f'5. Please set up your code to be run and save the results to the directory that its executed from \n6. Investigate grid search function')
print(f'Response: This question is ambiguous and it is unclear whether we are to \
inspect the example grid search in the hw or GridSearchCV by scikit-learn and \
whether we are to comment on it in any form. I will assume the latter was \
meant. The GridSearchCV function, despite being extensively documented \
only relies on two simple functions:\n')

print('def _run_search(self, evaluate_candidates)\: \n \"\"\"Search all candidates in param_grid\"\"\"\n evaluate_candidates(ParameterGrid(self.param_grid))\n')

print(f'In attempting to investigate these functions further, ParameterGrid function is previously classified whereby all combinations of parameters are made from grid.')