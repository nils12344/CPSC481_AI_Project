def insights_of_df(df):
    """
    Display of the insights of a data frame such as column names, number
    of null/non-null values, percentage null values, number of unique
    values and data types.ss

    Parameters
    ----------
        df: pandas.DataFrame.dtypes
            The full dataset

    Returns
    -------
        output: pandas.DataFrame.dtypes
            Dataframe with specified columns
    """
    import pandas as pd
    import numpy as np

    output = []

    # loop through all columns in the DataFrame
    for col in df.columns:
        nonNull  = len(df) - np.sum(pd.isna(df[col]))
        nullValues = np.sum(pd.isna(df[col]))
        percentNA = nullValues / (nullValues + nonNull)
        unique = df[col].nunique()
        colType = str(df[col].dtype)

        # Append variable to the output list
        output.append([col, nonNull, nullValues, percentNA, unique, colType])

    output = pd.DataFrame(output)
    output.columns = ['colName', 'non-null values', 'null values', 
                      'percentNA', 'unique', 'dtype']
    return output



def regression_metrics(y_true, y_pred):
    """
    Calculate the regression metrics such as mean absolute error, mean
    squared error, root mean squared error, mean absolute percentage
    error and R-squared.

    Parameters
    ----------
        y_true: pandas.Series
            The true values of the target variable
        y_pred: pandas.Series
            The predicted values of the target variable

    Returns
    -------
        output: 5 metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np

    # Calculate the metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    #mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)

    # return the metrics
    return mae, mse, rmse, r2