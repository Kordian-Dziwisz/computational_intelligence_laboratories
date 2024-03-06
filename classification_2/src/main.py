"""Organizing data"""

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree

import pandas as pd
# import numpy as np

# def parse_data(df: str):
#     df["sepal.length"]=df["sepal.length"].astype(float)

def get_data(path: str)->pd.DataFrame:
    """Read csv file at path"""

    return pd.read_csv(path)

def get_data_sets(df: pd.DataFrame)->(pd.DataFrame, pd.DataFrame):
    """create 2 data sets"""

    return train_test_split(df.values, train_size=0.7, random_state=13)

def get_inputs_and_classes(set_df: pd.DataFrame)->(pd.DataFrame, pd.DataFrame):
    """divide data set into input and  class"""
    return (
        set_df[:, 0:4],
        set_df[:, 4]
    )

def classify_iris(row)->str:
    """My method to classify iris"""
    if row[2] < 2:
        return "Setosa"
    elif row[0] > 7:
        return "Virginica"
    else:
        return "Versicolor"

def get_stats(df):
    vir_only = df[df["variety"]=="Virginica"]
    ver_only = df[df["variety"]=="Versicolor"]
    set_only = df[df["variety"]=="Setosa"]
    def get_single_stats(species_data):
        return {
        'sep_len_min': species_data["sepal.length"].min(),
        'sep_len_max': species_data["sepal.length"].max(),
        'sep_wid_min': species_data["sepal.width"].min(),
        'sep_wid_max': species_data["sepal.width"].max(),
        'pt_len_min': species_data["petal.length"].min(),
        'pt_len_max': species_data["petal.length"].max(),
        'pt_wid_min': species_data["petal.width"].min(),
        'pt_wid_max': species_data["petal.width"].max()
        }
    return {
        "Virginica": get_single_stats(vir_only),
        "Versicolor": get_single_stats(ver_only),
        "Setosa": get_single_stats(set_only)
    }

def get_model_accuracy(test_results, proper_results){
    proper = 0
    for i in range(len(test_results)):
        el1 = df.iloc[i].iloc[4]
        el2 = clf_results[i][0]
        if(el1==el2):
            proper += 1
    clf_result = proper/len(df)
}


def main()->None:
    """"Main function"""
    df = get_data(f"{Path.cwd()}/assets/iris.csv")
    (train_df, test_df) = get_data_sets(df)
    (train_df_inputs, train_df_classes) = get_inputs_and_classes(train_df)
    (test_df_inputs, test_df_classes) = get_inputs_and_classes(test_df)


    stats = get_stats(df)
    my_results = [classify_iris(row) for row in df.values] 
    print(get_model_accuracy(my_results, df))
    1 #show human results


    1 #show sets

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_df_inputs, train_df_classes)
    clf_results = [clf.predict([row[:4]]) for row in df.values] 
    print(get_model_accuracy(clf_results, train_df_classes))


    1 #show clf result

    tree.plot_tree(clf) #plot the tree, doesn't work
    



    




if __name__ == "__main__":
    main()
