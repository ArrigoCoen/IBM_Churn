
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


