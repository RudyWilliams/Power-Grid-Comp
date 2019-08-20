import dask
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor 
import timing_decorators as td 
import models


#create array for number of trees to use in the model
ntree_array = np.arange(0,801,100)
ntree_array[0] = 1 #i.e. Regression Tree

@td.runtime_timer
def run_CV(ntrees=1, m='sqrt', rand_state=None, output_type='raw_values'):
    """
        Run through a round of CV to get a CV error for the specified
        number of trees (ntrees)
    """
    rmses = np.empty(shape=(5,2))

    #5-Fold CV indicies
    kfold_5 = KFold(n_splits=5, shuffle=False, random_state=rand_state)
    for i, index_tuple in enumerate(kfold_5.split(models.X_weather)):
        train_index = index_tuple[0]
        test_index = index_tuple[1]
        X_train = models.X_weather.iloc[train_index,:]
        y_train = models.y_weather[train_index,:]
        X_test = models.X_weather.iloc[test_index,:]
        y_test = models.y_weather[test_index,:]
        
        rfr = RandomForestRegressor(
            n_estimators=ntrees,
            max_features=m,
            min_samples_leaf=1,
            oob_score=False,
            random_state=rand_state
        )
        rfr.fit(X_train, y_train)
        fold_pred = rfr.predict(X_test)
        fold_mse = mean_squared_error(y_test, fold_pred, multioutput=output_type)
        fold_rmse = np.sqrt(fold_mse)
        rmses[i] = fold_rmse

    return rmses.mean(axis=0)

#good candidate for dask
@dask.delayed
def delayed_fit_eval(model, X_train, y_train, X_test, y_test):
    """
        Delay the actual fitting and evaluation of given model. This
        gets called for each fold. This might be going against a best 
        practice...
    """
    model.fit(X_train, y_train)
    y_preds = model.predict(X_test)
    fold_mse = mean_squared_error(y_test, y_preds, multioutput='raw_values')
    fold_rmse = np.sqrt(fold_mse)

    return fold_rmse

@td.runtime_timer
def run_parallel_CV(ntrees=1, m='sqrt', rand_state=None, output_type='raw_values'):
    """
        Run through a round of CV to get a CV error for the specified
        number of trees (ntrees)
    """
    rmses = [0]*5

    #5-Fold CV indicies

    kfold_5 = KFold(n_splits=5, shuffle=False, random_state=rand_state)
    for i, index_tuple in enumerate(kfold_5.split(models.X_weather)):
        train_index = index_tuple[0]
        test_index = index_tuple[1]
        X_train = models.X_weather.iloc[train_index,:]
        y_train = models.y_weather[train_index,:]
        X_test = models.X_weather.iloc[test_index,:]
        y_test = models.y_weather[test_index,:]
        
        #have to reinstantiate to reset rfr for each fold
        rfr = RandomForestRegressor(
            n_estimators=ntrees,
            max_features=m,
            min_samples_leaf=1,
            oob_score=False,
            random_state=rand_state
        )

        fold_errors = delayed_fit_eval(
            model=rfr,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )

        rmses[i] = fold_errors

    #after all CV folds are complete
    rmses_array = np.array(dask.compute(*rmses)) #recall rmses is a list of delayed objects

    return rmses_array.mean(axis=0)
    



if __name__ == '__main__':

    print(run_parallel_CV(ntrees=500, m='sqrt', rand_state=42, output_type='raw_values'))
    # ntrees_cv_errors = [run_CV(ntrees=n, m='sqrt',rand_state=42) for n in ntree_array]
    # print(ntrees_cv_errors)
    print(run_CV(ntrees=500, rand_state=42))
    pass

