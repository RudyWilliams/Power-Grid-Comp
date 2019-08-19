from math import sqrt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.linear_model import LinearRegression, Lasso 
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
import data_prep as dp


#first model will run linear regression
def train_linear_regression(X_data, y_data, split=False, train_pct=None, rand_state=None):
    """
    Fit a linear regression model and return the model. If split=True,
    return the model and the validation MSE. Note, this model ignores the
    time series aspect of the data.
    
    Args:
        X_data: pd.DataFrame - the feature matrix
        y_data: array (n_samples, n_targets)- the target array
        split: bool (default: False) - Train and validate if True or just 
               train if False
        train_pct: float btwn 0 & 1 - signifies the train test split size
        rand_state: int (default: None) - to create reproducible results
    
        Returns:
            (linreg, y_pred)
            linreg: sklearn LinearRegression model - the model resulting from the fit on
                    either the full dataset (split=False) or fit on the training data
                    within the function (split=True)
    
                or
            (linreg, mse, rmse) tuple if split=True
            mse: float or array of shape (n_targets,) - the mse calculated if split=True
            rmse: float or array of shape (n_targets,) - the root mse calculated if split=True

    """
    if split:
        X_train, X_test, y_train, y_test = train_test_split(
            X_data, 
            y_data, 
            random_state=rand_state,
            train_size=train_pct
            )

        linreg = LinearRegression(n_jobs=-1)
        linreg.fit(X_train, y_train)
        y_pred = linreg.predict(X_test)

        #compute MSE using mean_squared_error
        #by default multi_output='uniform_average' which is why it was returning a single value
        mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
        rmse = np.sqrt(mse)

        return linreg, mse, rmse

    else:
        linreg = LinearRegression(n_jobs=-1)
        linreg.fit(X_data, y_data)
      
        return linreg

    
def train_lasso_regression(X_data, y_data, alpha=1, split=False, cross_val=False, folds=5, train_pct=None, rand_state=None):
    """
    """
    if split and cross_val:
        raise ValueError('Args split and cv cannot both be set to True.')

    elif split: #single train and test run
        X_train, X_test, y_train, y_test = train_test_split(
            X_data,
            y_data, 
            random_state=rand_state,
            train_size=train_pct
            )

        lasso = Lasso(random_state=rand_state, alpha=alpha, tol=0.01)
        lasso.fit(X_train, y_train)
        y_pred = lasso.predict(X_test)
        mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
        rmse = np.sqrt(mse)

        return lasso, mse, rmse
    
    elif cross_val:
        lasso = Lasso(random_state=rand_state, alpha=alpha)
        mse_scorer = make_scorer(mean_squared_error, multioutput='uniform_average', greater_is_better=False)
        #a scalar must be returned, not an array
        cv_scores = cross_validate(lasso, X_data, y_data, cv=folds, scoring=mse_scorer)
        return cv_scores
        
    else:
        pass


def train_random_forest_regression(X_data, y_data, split=False, train_pct=None, rand_state=None,
                                 ntrees=1, min_samples_node=1):
    """
    Fit a linear regression model and return the model. If split=True,
    return the model and the validation MSE. Note, this model ignores the
    time series aspect of the data.
    
    Args:
        X_data: pd.DataFrame - the feature matrix
        y_data: array (n_samples, n_targets)- the target array
        split: bool (default: False) - Train and validate if True or just 
               train if False
        train_pct: float btwn 0 & 1 - signifies the train test split size
        rand_state: int (default: None) - to create reproducible results
    
        Returns:
    """
    if split:

        X_train, X_test, y_train, y_test = train_test_split(
            X_data,
            y_data, 
            random_state=rand_state,
            train_size=train_pct
            )

        #above should produce the same sets given the same rand_state is given
        rfr = RandomForestRegressor(
            n_estimators=ntrees,
            max_features='sqrt',
            min_samples_leaf=min_samples_node,
            random_state=rand_state,
            oob_score=False
        )

        rfr.fit(X_train, y_train)
        y_pred = rfr.predict(X_test)
        mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
        rmse = np.sqrt(mse)

        return rfr, mse, rmse

    #use out-of-bag error
    else:    
    
        rfr = RandomForestRegressor(n_estimators=ntrees, max_features='sqrt',
                                    min_samples_leaf=min_samples_node,
                                    random_state=rand_state, oob_score=True)
        rfr.fit(X_data, y_data)
        y_oob_pred = rfr.oob_prediction_  #should maintain order
        oob_mse = mean_squared_error(y_data, y_oob_pred, multioutput='raw_values') #use all of y_data (no splitting)
        oob_rmse = np.sqrt(oob_mse)

        return rfr, oob_mse, oob_rmse



if __name__ == '__main__':

    from submit_model import upload_response

    ### Read training data without weather data
    #+++++++++++++++++++++++++++++++++++++++++++
    X, y = dp.read_training(index_columns=['date'], both=True)

    X_feed = dp.ffill_nans(X)
    y_feed = y.loc[:,['target1','target2']].values
    
    ### Read test data without weather data
    #+++++++++++++++++++++++++++++++++++++++++
    X_test = dp.read_test_X(index_columns=['date'])
    X_test = dp.ffill_nans(X_test)

    ### Read training data with weather data
    #+++++++++++++++++++++++++++++++++++++++++
    X_weather, y_weather = dp.read_training(
        both=True, 
        weather=True, 
        index_columns=['date']
        )

    X_weather = dp.fill_weather_forecast_columns(X_weather)
    y_weather = y_weather.loc[:,['target1','target2']].values
    
    ### Read test data with weather data
    #+++++++++++++++++++++++++++++++++++++++
    X_test_weather = dp.read_test_X(index_columns=['date'], weather=True)
    X_test_weather = dp.fill_test_weather_forecast_columns(X_test_weather)
    
    ######################################################

    ## perform linear regression without weather data
    #================================================
    model, mse, rmse = train_linear_regression(
        X_feed,
        y_feed,
        split=True,
        train_pct=0.8, 
        rand_state=5
        )
    
    print(f'Linear Regression RMSE: {rmse}')

    ## submit linear regression model
    #----------------------------------------
    linear_regression = train_linear_regression(X_feed, y_feed, split=False)
    turn_in_pred = linear_regression.predict(X_test)
    print(f'  -> Shape of submission array: {turn_in_pred.shape}')
    # upload_response('RW-LinearRegression', turn_in_pred)

    ## perform Linear Regression with the weather data
    #==================================================
    lr_modelW, mseW, rmseW = train_linear_regression(
        X_weather,
        y_weather,
        split=True,
        train_pct=0.8,
        rand_state=5
        )

    print(f'Linear Regression w/ Weather Data RMSE: {rmseW}')    

    ## submit linear regression w/ weather data
    #--------------------------------------------
    trained_lr_weather = train_linear_regression(X_weather, y_weather)
    lrW_pred_turn_in = trained_lr_weather.predict(X_test_weather)
    print(f'  -> Shape of sumbission array: {lrW_pred_turn_in.shape}')
    # upload_response('RW-LinRegWithWeather', lrW_pred_turn_in)
    #Rank 1 as of 8/19/2019 @2:10pm


    ## perform Random Forest Regression w/ Weather Data
    #===================================================
    rfr_results = train_random_forest_regression(
        X_data=X_weather,
        y_data=y_weather,
        ntrees=800,
        split=True,
        train_pct=0.8,
        rand_state=5
    )
    print(f'Random Forest Regressor w/ Weather Data RMSE: {rfr_results[2]}')

    ### submit random forest regressor model w/ weather data
    #--------------------------------------------------------
    trained_rfr, _, oob_rmse = train_random_forest_regression(
        X_weather,
        y_weather,
        ntrees=800,
        rand_state=5
    )
    print(f'Out-of-Observation RMSE: {oob_rmse}')
    rfr_pred_turn_in = trained_rfr.predict(X_test_weather)
    print(f'  -> Shape of submission array: {rfr_pred_turn_in.shape}')

    ## perform Lasso Regression with the weather data
    # split_results = train_lasso_regression(
    #     X_weather,
    #     y_weather,
    #     alpha=0.01,
    #     split=True,
    #     train_pct=0.8,
    #     rand_state=5
    #     )

    ##gets a does not converge warning even after playing with max_iter and tol a bit
    # print(f'Lasso w/ Weather Data (alpha=XXXXX) RMSE: {split_results[2]}')

    ## perform Lasso regression without weather data
    #===============================================
    # split_results = train_lasso_regression(
    #     X_feed,
    #     y_feed,
    #     alpha=0.1,
    #     split=True,
    #     train_pct=0.8,
    #     rand_state=5
    #     )
    
    # print(f'Lasso RMSE (alpha=0.1): {split_results[2]}')
