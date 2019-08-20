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
def run_CV(model, nfold, X, y, rand_state=None, output_type='raw_values', **kwargs):
    """
        Run through a round of CV to get a CV error for the specified
        number of trees (n_estimators). This function is run sequentially.

        Args:
            model: sklearn model
            nfold: int - the number of folds for cross-validation
            X: pd.DataFrame - feature frame,
            y: np.array - target array, shape (n_obs, n_targets)
            rand_state: int - to create reproducible/comparable results
            output_type: str (default: 'raw_values) - either 'uniform_average'
                or 'raw_values', how mean_squared_error deals with multiple targets
            **kwargs: keyword args for the sklearn model
                e.g. n_estimators=100 for a random forest

        Returns:
            rmses.mean(axis=0): the cross-validated rmse error (i.e. the mean of the
                rmses calculated at each fold for each target)
    """
    rmses = np.empty(shape=(5,2))

    #nfold-Cross-Validation
    kfold = KFold(n_splits=nfold, shuffle=False, random_state=rand_state)
    for i, index_tuple in enumerate(kfold.split(X)):
        train_index = index_tuple[0]
        # print(train_index)
        test_index = index_tuple[1]
        # print(test_index)
        X_train = X.iloc[train_index,:]
        y_train = y[train_index,:]
        X_test = X.iloc[test_index,:]
        y_test = y[test_index,:]
        
        instan_model = model(random_state=rand_state, **kwargs)
        instan_model.fit(X_train, y_train)
        fold_pred = instan_model.predict(X_test)
        fold_mse = mean_squared_error(y_test, fold_pred, multioutput=output_type)
        fold_rmse = np.sqrt(fold_mse)
        rmses[i] = fold_rmse

    return rmses.mean(axis=0)


@td.runtime_timer
def run_rfr_CV(ntrees=1, m='sqrt', rand_state=None, output_type='raw_values'):
    """
        Run through a round of CV to get a CV error for the specified
        number of trees (ntrees). 
        * The specific function for random forest regression. Equivalent to 
          run_CV(model=RandomForestRegressor, ...)
    """
    rmses = np.empty(shape=(5,2))
    
    #5-Fold CV indicies
    kfold_5 = KFold(n_splits=5, shuffle=False, random_state=rand_state)
    for i, index_tuple in enumerate(kfold_5.split(models.X_weather)):
        train_index = index_tuple[0]
        # print(train_index)
        test_index = index_tuple[1]
        # print(test_index)
        X_train = models.X_weather.iloc[train_index,:]
        y_train = models.y_weather[train_index,:]
        X_test = models.X_weather.iloc[test_index,:]
        y_test = models.y_weather[test_index,:]
        
        #must be done for each iteration to reset
        rfr = RandomForestRegressor(
            n_estimators=ntrees,
            max_features=m,
            min_samples_leaf=1,
            random_state=rand_state
        )
        rfr.fit(X_train, y_train)
        fold_pred = rfr.predict(X_test)
        fold_mse = mean_squared_error(y_test, fold_pred, multioutput=output_type)
        fold_rmse = np.sqrt(fold_mse)
        rmses[i] = fold_rmse

    return rmses.mean(axis=0)


#for loop w/ indep iterations => good candidate for dask
@dask.delayed
def delayed_fit_eval(model, X_train, y_train, X_test, y_test):
    """
        Delay the actual fitting and evaluation of given model. This
        gets called for each fold.

        Args:
            model: sklearn model - passed in from parent function run_parallel_CV
            X_train: pd.DataFrame - the training data created in run_parallel_CV 
                by indexing X with the training indicies of the current fold
            y_train: np.array - the targets for the training
            X_test: pd.DataFrame - the test data for current fold
            y_test: np.array - the targets for the testing so that the rmse
                can be calculated

        Returns:
            fold_rmse: dask delayed object - gets computed in run_parallel_CV
    """
    model.fit(X_train, y_train)
    y_preds = model.predict(X_test)
    fold_mse = mean_squared_error(y_test, y_preds, multioutput='raw_values')
    fold_rmse = np.sqrt(fold_mse)

    return fold_rmse


@td.runtime_timer
def run_parallel_CV(model, nfold, X, y, rand_state=None, output_type='raw_values', **kwargs):
    """
        Run through a round of CV to get a CV error for the specified
        tuning parameter (e.g. n_estimators). This function uses dask.delayed
        to fit and evaluate the model at each fold in parallel.

        Args:
            model: sklearn model
            nfold: int - the number of folds for cross-validation
            X: pd.DataFrame - feature frame,
            y: np.array - target array, shape (n_obs, n_targets)
            rand_state: int - to create reproducible/comparable results
            output_type: str (default: 'raw_values) - either 'uniform_average'
                or 'raw_values', how mean_squared_error deals with multiple targets
            **kwargs: keyword args for the sklearn model
                e.g. n_estimators=100 for a random forest

        Returns:
            rmses.mean(axis=0): the cross-validated rmse error (i.e. the mean of the
                rmses calculated at each fold for each target)

    """
    rmses = [0]*5

    #nfold-Cross-Validation
    kfold = KFold(n_splits=nfold, shuffle=False, random_state=rand_state)
    for i, index_tuple in enumerate(kfold.split(X)):
        train_index = index_tuple[0]
        # print(train_index)
        test_index = index_tuple[1]
        # print(test_index)
        X_train = X.iloc[train_index,:]
        y_train = y[train_index,:]
        X_test = X.iloc[test_index,:]
        y_test = y[test_index,:]
        
        #have to reinstantiate to reset model for each fold
        instan_model = model(random_state=rand_state, **kwargs)

        fold_errors = delayed_fit_eval(
            model=instan_model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )
        rmses[i] = fold_errors

    #after all CV folds are complete
    rmses_array = np.array(dask.compute(*rmses)) #recall rmses is a list of delayed objects

    return rmses_array.mean(axis=0)


@td.runtime_timer
def run_parallel_rfr_CV(ntrees=1, m='sqrt', rand_state=None, output_type='raw_values'):
    """
        Run through a round of CV to get a CV error for the specified
        number of trees (ntrees). Done in parallel.
        * Equivalent to run_parallel_rfr_CV(model=RandomForestRegressor, ...)
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

    # print(run_parallel_CV(ntrees=500, m='sqrt', rand_state=42, output_type='raw_values'))
    # ntrees_cv_errors = [run_CV(ntrees=n, m='sqrt',rand_state=42) for n in ntree_array]
    # print(ntrees_cv_errors)
    print(run_CV(
        RandomForestRegressor,
        5, 
        models.X_weather, 
        models.y_weather,
        rand_state=42,
        n_estimators=1,
        max_features='sqrt', #was forgetting to set which caused longer runtime and bagging
        min_samples_leaf=1
    ))

    print(run_rfr_CV(
        ntrees=1,
        rand_state=42
    ))

    print(run_parallel_CV(
        RandomForestRegressor,
        5,
        models.X_weather,
        models.y_weather,
        rand_state=42,
        n_estimators=100,
        max_features='sqrt',
        min_samples_leaf=1
    ))

    print(run_parallel_rfr_CV(
        ntrees=100,
        rand_state=42,
        output_type='raw_values',
        m='sqrt'
    ))

    print(run_CV(
        RandomForestRegressor,
        5, 
        models.X_weather, 
        models.y_weather,
        rand_state=42,
        n_estimators=100,
        max_features='sqrt', #was forgetting to set which caused longer runtime and bagging
        min_samples_leaf=1
    ))
    

