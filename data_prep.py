"""
This will probably become more important if/when cleaning the
weather data.
"""

import pandas as pd 
from eda import count_column_nans

def read_training(index_columns=None, both=False, weather=False):
    """
    read the training X and y (optional) files
    
    Args:
        index_cols: list - the column name(s) to index on

        both: bool - whether or not to read in the target file too
        weather: bool - if True the weather data file will be read and
                 merged to the training data
    Returns:
        raw_X_train: pd.DataFrame - the data with only prepocessing being
                     completed at this stage being what should be parsed as
                     dates and what is the index column or the data merged with
                     the weather data. If this is the case then X ffills the nan
                     values to make the focus be on filling the observations without
                     weather data.
            or
        raw_X_train, raw_y_train: tuple of pd.DataFrames - returned when both=True
    """
    if weather:
        raw_X_train = pd.read_csv('data\\train_X.csv', parse_dates=['date'])
        raw_weather = pd.read_csv('data\\weather_data.csv', parse_dates=['date'])

        raw_X_train = ffill_nans(raw_X_train)
        raw_X_train = raw_X_train.merge(raw_weather, how='left', on=['date','hour'])
        raw_X_train = raw_X_train.set_index(index_columns)

    else:
        raw_X_train = pd.read_csv(
            'data\\train_X.csv',
             parse_dates=['date'],
             index_col=index_columns)
    if both:
        raw_y_train = pd.read_csv(
            'data\\train_y.csv',
            parse_dates=['date'],
            index_col=index_columns)

        return raw_X_train, raw_y_train
    
    return raw_X_train


def read_test_X(index_columns='warn', weather=False):
    """
    Read in the test data for the competition. There is no target vector y. The index
    columns need to be the same as the training data.
    """
    if index_columns=='warn':
        print('Make sure index_columns is set to same column(s) as in read_training.\
            Reading in test_X.csv w/o any index column(s) specified.')
        index_columns=None
    
    if weather:
        raw_X_test = pd.read_csv('data\\test_X.csv', parse_dates=['date'])
        raw_weather = pd.read_csv('data\\weather_data.csv', parse_dates=['date'])

        raw_X_test = ffill_nans(raw_X_test)
        raw_X_test = raw_X_test.merge(raw_weather, how='left', on=['date','hour'])
        X_test = raw_X_test.set_index(index_columns)
    
    else:
        X_test = pd.read_csv('data\\test_X.csv', parse_dates=['date'], index_col=index_columns)    
    
    return X_test


#if wanting to deal with the data together
def create_Xy_df(X_df, y_df, on_cols):
    """
    I like doing the data cleaning process with the data as
    a whole. That way if any rows are dropped (which won't happen
    in this competition) the target rows are also dropped. Perhaps
    it's better to leave them split though and just pass the index
    of the X to filter the y. In this way I won't risk accidentally
    transforming the values in the target.
    """
    return pd.merge(X_df, y_df, how='inner', on=on_cols)

def ffill_nans(df):
    """
    Data is expected to be in order by date and hour (otherwise
    use sort on those two columns before hand). This replaces the
    NaN with the value, of the same feature, from the preceeding 
    observation.
    """
    return df.fillna(method='ffill', axis=0)


def fill_weather_forecast_columns(df):
    """
        ffill, bfill, ffill should fill them appropriately. First, Jan. 1st needs all hours
        data. To do this we will use Jan. 2nd numbers. This doesn't seem to produce a smooth
        fix as the number for wind_north_OK goes from 2.11 to -2.44 but it's a start.

        I will set limits on the ffill/bfill so that it only fills the NaN of the hour that
        it 'touches'.

        Args:
            df: pd.DataFrame - the dataframe that needs filling
        
        Returns:
            filled_df: pd.DataFrame - the dataframe that has the nan's filled
    """

    filled_df = df.copy()
    filled_df.loc['2018-01-01','temp_KC':'wind_north_SD'] = filled_df.loc['2018-01-02','temp_KC':'wind_north_SD'].values
    filled_df.loc['2018-02-06','temp_KC':'wind_north_SD'] = filled_df.loc['2018-02-05','temp_KC':'wind_north_SD'].values
    filled_df.loc['2019-02-05','temp_KC':'wind_north_SD'] = filled_df.loc['2019-02-04','temp_KC':'wind_north_SD'].values
    # print(filled_df.isna().sum(axis=0))
    filled_df = filled_df.fillna(method='ffill', limit=1)
    # print(filled_df.isna().sum(axis=0))
    filled_df = filled_df.fillna(method='bfill', limit=1)
    # print(filled_df.isna().sum(axis=0))
    filled_df = filled_df.fillna(method='ffill', limit=1)

    any_nans = filled_df.isna().sum(axis=0)
    
    if any_nans.sum(axis=0) != 0:
        print('The function did not convert all NaNs. Some NaNs still exist.')

    return filled_df

def fill_test_weather_forecast_columns(df):
    """
        these fill functions are very specific to the data at hand. Hope to make more
        general to deal with missing data (in chunks of days) without user direction.

        Args:
            df: pd.DataFrame - the dataframe that needs filling
        
        Returns:
            filled_df: pd.DataFrame - the dataframe that has the nan's filled
    """

    filled_df = df.copy()
    filled_df.loc['2018-02-22','temp_KC':'wind_north_SD'] = filled_df.loc['2018-02-21','temp_KC':'wind_north_SD'].values
    filled_df.loc['2018-02-23','temp_KC':'wind_north_SD'] = filled_df.loc['2018-02-24','temp_KC':'wind_north_SD'].values
    # print(filled_df.loc['2018-02-21':'2018-02-24'])
    filled_df = filled_df.fillna(method='ffill', limit=1)
    # print(filled_df.isna().sum(axis=0))
    filled_df = filled_df.fillna(method='bfill', limit=1)
    # print(filled_df.isna().sum(axis=0))
    filled_df = filled_df.fillna(method='ffill', limit=1)

    any_nans = filled_df.isna().sum(axis=0)
    
    if any_nans.sum(axis=0) != 0:
        print('The function did not convert all NaNs. Some NaNs still exist.')

    return filled_df
        
if __name__ == '__main__':

    pass
    #w/ weather data
    #----------------
    # XW_train, y_train = read_training(index_columns=['date'], both=True, weather=True)

    # XW_train = fill_weather_forecast_columns(XW_train)
    # print(XW_train.info())
    # print(y_train.info())
    # XW_train.to_csv('data\\XW_train.csv')

    # X = XW_train.copy()
    # print(X.loc['2018-01-01':'2018-01-02', 'temp_KC':'wind_north_SD']) #2018-01-01 needs all the data
    # X.loc['2018-01-01', 'temp_KC':'wind_north_SD'] = X.loc['2018-01-02', 'temp_KC':'wind_north_SD'].values
    # print(X.loc['2018-01-01':'2018-01-02', 'coal':'wind_north_SD'])
    # example = XW_train.loc['2018-07-10':'2018-07-11',['hour','wind_north_OK']] 
    # print(example.isna().sum(axis=0))
    # print(example.isna().sum(axis=0).sum(axis=0))
    # example = example.fillna(method='ffill', axis=0, limit=1)
    # print(example)
    # example = example.fillna(method='bfill', axis=0, limit=1)
    # print(example)
    # example = example.fillna(method='ffill', axis=0, limit=1)
    # print(example)
    #running the model still works as expected when no weather data
    


    #w/o weather data
    #-----------------
    # X_train = read_training(index_columns=['date'], both=False)
    # #fill na by ffill
    # print(count_column_nans(X_train))
    # X_train_ffilled = ffill_nans(X_train)
    # print(count_column_nans(X_train_ffilled))

    # #send to csv
    # X_train_ffilled.to_csv('data\\ffilled_X_train.csv')

    # X_test = read_test_X(index_columns=['date'])
    # print(X_test.head())