import numpy as np
import timeit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

def Modelling(estimator, X_train, y_train):
    # Function to fit an estimator with required dataset.
    start = timeit.default_timer()
    estimator.fit(X_train, y_train)
    end = timeit.default_timer() - start
    return estimator, end

def optimal_model(model, grid, X_train, y_train):
    # Function used to execute GridSearchCV on a specific estimator with param_grids
    grid_search = GridSearchCV(estimator=model, param_grid=grid, cv=3, verbose=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_, grid_search.cv_results_

# def KNN(X_train, y_train):
#     knn_model = KNeighborsClassifier()
#     param_grid = {
#         'n_neighbors': [100, 250, 500],
#         'weights': ['distance'],
#     }
#     optimal_knn_params, results = optimal_model(knn_model, param_grid, X_train, y_train)
#     return optimal_knn_params, results

def Decision_Tree(X_train, y_train):
    # Function that find the optimal parameters using GridSearchCV for a Decision Tree
    dt_model = DecisionTreeClassifier()
    param_grid = {
        'max_depth': np.arange(5, 100, 10),
        'min_samples_split': [2, 10, 50, 100, 500],
        'min_samples_leaf': [1, 5, 10, 50, 100],
        'max_features': ['sqrt', 'log2']
    }
    optimal_dt_params, results = optimal_model(dt_model, param_grid, X_train, y_train)
    return optimal_dt_params, results

# def NeuralNetwork(X_train, y_train):
#     nn_model = MLPClassifier()
#     param_grid = {
#         'hidden_layer_sizes': [(50,), (100,)],
#         'activation': 'relu',
#         'solver': 'sgd',
#         'alpha': [0.0001, 0.05],
#         'learning_rate': ['constant', 'adaptive'],
#         'max_iter': [200, 400, 800]
#     }
#     optimal_nn_params, results = optimal_model(nn_model, param_grid, X_train, y_train)
#     return optimal_nn_params







