
def my_piclke_dump(var, file_name):
    """
    This function saves 'var' in a pickle file with the name 'file_name'.
    We use the folder 'Pickles' to save all the pickles.
    Example:

    file_name = "test"
    x = 4
    var = x
    my_piclke_dump(var, file_name)

    :param var: variable to save
    :param file_name: file name
    :return: True
    """
    from pathlib import Path
    import pickle
    file_path = Path().joinpath('Pickles', file_name + ".pkl")
    pickle_out = open(file_path, "wb")
    pickle.dump(var, pickle_out)
    pickle_out.close()
    print("The file ", file_path, "was save.")
    return(True)


def my_piclke_load(file_name):
    """
    General extraction of variables from a pickle file.
    Example:

    file_name = "test"
    x = 4
    var = x
    my_piclke_dump(var, file_name)
    zz = my_piclke_load(file_name)

    :param file_name: name of the pickle file
    :return: the variable inside the file
    """
    import pandas as pd
    file_path = Path().joinpath('Pickles', file_name + ".pkl")
    var = pd.read_pickle(file_path)
    print("The file ", file_name, ".pkl was loaded.")
    return var


def model_results(model, results_df, model_name=None, verbose=0):
    """
    Given a model this function updats the dataframe results_df with the model's cross-validation (CV) results. This
    function also plots the errors with respect to the X_test and y_test sets.
    :param model: a machine learning model; eg. LinearRegression()
    :param results_df: a pd.DataFrame with the current CV information of the models
    :param model_name: an extra string with the name that will be use the information of the model. This variables
    is useful in case of having different instances of the model with different parameters. In case of None, it
    uses type(model).__name__
    :return: an update version of results_df with the `model`'s information
    """
    # Name of the model
    if model_name is None:
        model_name = type(model).__name__
    print(model_name)
    # Pipeline to do a cross validation
    pipe = make_pipeline(column_trans, model)
    # Results of the CV
    c_scores = cross_validate(pipe, X_train, y_train, cv=5,
                              scoring=['neg_mean_absolute_error', 'neg_mean_squared_error'],
                              verbose=verbose, n_jobs=-1)
    # Extracting CV results
    mse_mean = np.sqrt(np.array(-c_scores['test_neg_mean_squared_error'])).mean()
    mse_std = c_scores['test_neg_mean_squared_error'].std()
    mae_mean =  np.sqrt(np.array(-c_scores['test_neg_mean_absolute_error'])).mean()
    mae_std = c_scores['test_neg_mean_absolute_error'].std()
    fit_time_mean = c_scores['fit_time'].mean()
    # We define the new row of results_df
    new_row = [model_name, mse_mean, mse_std, mae_mean, mae_std]
    # Plot of predictions
    pipe.fit(X_train, y_train)

    my_piclke_dump(pipe, "Model_"+model_name)

    y_pred = pipe.predict(X_test)
    plt.plot(np.array(y_test)-y_pred)
    plt.title("mse of predictions of"+model_name)
    plt.show()
    # This if take cares of the case when resuls_df is empty
    if results_df.iloc[0,0]== 0:
        results_df.loc[0] = new_row
    else:
        results_df.loc[len(results_df.index)] = new_row
    return results_df

def normalized_bar_plot(results_df):
    """
    Bar plot of the normalized erros
    :param results_df: data frame with the model`s cross-validation errors.
    :return:
    """
    normalized_df = results_df.copy()
    normalized_df.iloc[:,1:]=(results_df.iloc[:,1:].copy())/results_df.iloc[:,1:].max()
    normalized_df.plot(x="model", y=['rmse_mean', 'rmse_std', 'mae_mean', 'mae_std'], kind="bar")
    # plt.ylim([0,2])
    plt.title("Equivalent magnitudes of results")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

def test_my_print():
    print("functionaaaaa!")
    return True


