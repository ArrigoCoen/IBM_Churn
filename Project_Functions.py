"""
modulereload(Project_Functions)
from Project_Functions import *



# modulereload(Project_Functions)
# from Project_Functions import *

"""




def modulereload(modulename):
    """
    Use:
        modulereload(Project_Functions)
        from Project_Functions import *
    :param modulename:
    :return:
    """
    # https://stackoverflow.com/questions/18500283/how-do-you-reload-a-module-in-python-version-3-3-2
    import importlib
    importlib.reload(modulename)

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
    return True


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
    from pathlib import Path
    file_path = Path().joinpath('Pickles', file_name + ".pkl")
    var = pd.read_pickle(file_path)
    print("The file ", file_name, ".pkl was loaded.")
    return var


def my_print_test32():
    print("OK")
    return True



def model_results(model, results_df, column_trans, X_train, X_test, y_train, y_test, model_name=None, my_verbose=0,
                  save_as_network=False):
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
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_validate
    from sklearn.metrics import classification_report, confusion_matrix

    # Name of the model
    if model_name is None:
        model_name = type(model).__name__
    print(model_name)

    my_score = ['accuracy', 'f1_weighted']

    pipe = make_pipeline(column_trans, model)
    # Results of the CV
    c_scores = cross_validate(pipe, X_train, y_train, cv=5,scoring=my_score,
                              verbose=my_verbose, n_jobs=-1)

    accuracy_mean = c_scores['test_accuracy'].mean()
    accuracy_std = c_scores['test_accuracy'].std()
    f1_mean = c_scores['test_f1_weighted'].mean()
    f1_std = c_scores['test_f1_weighted'].std()

    new_row = [model_name, accuracy_mean, accuracy_std, f1_mean, f1_std]
    # print(c_scores)
    # Prediction using all the train examples
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    # We save is pickle if is any other model except a keras Sequential
    if save_as_network:
        # save_NN_sequential(pipe, model_name)
        # my_piclke_dump(model, "Model_"+model_name)
        print("")
    else:
        my_piclke_dump(pipe, "Model_"+model_name)
    cf_matrix = confusion_matrix(y_test,y_pred)
    plot_cf_matrix(cf_matrix)
    print(classification_report(y_test,y_pred))
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
    import matplotlib.pyplot as plt
    normalized_df = results_df.copy()
    normalized_df.iloc[:,1:]=(results_df.iloc[:,1:].copy())/results_df.iloc[:,1:].max()
    normalized_df.plot(x=results_df.columns[0], y=results_df.columns[1:], kind="bar")
    plt.title("Equivalent magnitudes of results")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()



def col_vr_target(df, target_name, not_to_plot):
    """
    Plot of each column vr target column
    :param df: dataframe to plot
    :param target_name: name of target column as string
    :param not_to_plot: Columns that for some reason should not be ploted
    :return:
    """

    import matplotlib.pyplot as plt
    import seaborn as sns
    my_palette = ['colorblind', 'deep', 'pink', 'magma'][0]

    # dataframe['Columnn name'].value_counts()
    for col in df.columns:
        print('column = ', col)
        print(df[col].dtypes)
        if df[col].dtypes in ['int64', 'float64']:
            print(df[col].describe())
            if col not in  [target_name] + not_to_plot:
                sns.catplot(x=target_name,y=col, kind="violin", data=df, palette = my_palette)
                plt.title(target_name+' vr '+col)
                plt.show()
        if df[col].dtypes == 'object':
            print(df[col].value_counts())
            if col not in [target_name] + not_to_plot:
                sns.catplot(x=col, kind="count", hue = target_name, palette=my_palette, data=df)
                plt.title(target_name + ' vr '+ col)
                plt.show()

    return True


def plot_cf_matrix(cf_matrix):
    """
    Plot of the confusion matrix
    :param cf_matrix:
    :return:
    """

    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt

    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']

    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in
                         cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names, group_counts, group_percentages)]

    labels = np.asarray(labels).reshape(2, 2)

    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')

    ax.set_title('Seaborn Confusion Matrix with labels\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])

    ## Display the visualization of the Confusion Matrix.
    plt.show()


def save_NN_sequential(model, model_name):
    """
    Saving a Neural Network as h5 file
    :param model: sequential model
    :param model_name: name to save the model
    :return: True
    """
    from pathlib import Path
    file_name = 'Model_' + model_name
    file_path = Path().joinpath('Pickles', file_name + ".h5")
    print("The file ", file_path, "was save.")
    model.save(file_path)
    return True
