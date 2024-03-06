"""Organizing data"""

from pathlib import Path
from sklearn.model_selection import train_test_split

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

def main()->None:
    """"Main function"""
    df = get_data(f"{Path.cwd()}/assets/iris_with_errors.csv")

    df = pd.read_csv(f"{Path.cwd()}/assets/iris_with_errors.csv")

    # print(df.head())
    # print_empty_fields(df)
    fix_ranges(df)


if __name__ == "__main__":
    main()
