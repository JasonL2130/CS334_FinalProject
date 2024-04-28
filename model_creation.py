import numpy as np
import timeit
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

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

def XGBoost(X_train, y_train):
    # Function that finds the optimal parameters using GridSearchCV for an XGBoost model
    xgb_model = XGBClassifier()
    param_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'subsample': [0.7, 0.8, 0.9]
    }
    optimal_xgb_params, results = optimal_model(xgb_model, param_grid, X_train, y_train)
    return optimal_xgb_params, results






