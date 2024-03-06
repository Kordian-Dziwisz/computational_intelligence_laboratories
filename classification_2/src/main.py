"""Organizing data"""

from pathlib import Path

import pandas as pd
# import numpy as np

# def parse_data(df: str):
#     df["sepal.length"]=df["sepal.length"].astype(float)

def get_data(path: str)->pd.DataFrame:
    """Read csv file at path"""

    return pd.read_csv(path)

def print_empty_fields(df: pd.DataFrame) -> None:
    """Prints missing fields summary to std"""
    print(df.isnull().sum())


def fix_ranges(df: pd.DataFrame) -> None:
    """Fixes dataframe columns"""

    def get_without_empty_rows(column_names: list) -> pd.DataFrame:
        """deletes "empty" rows from specific dataframe columns"""
        for column_name in column_names:
            final_df = df[df[column_name] != '-']
        return final_df
    
    def get_average(column_name: str) -> float:
        """get avg of a specific colummn"""
        return df[column_name].astype(float).avg()

            avg = df[(df[column_name] != "-") and (df[column_name] > 0) and (df[column_name] < 15)][column_name].astype(float)
            df[df[column_name] == "-" or df[column_name] <= 0 or df[column_name] >= 15][column_name]=avg

    
    fix_columns(["sepal.length", "sepal.width", "petal.length", "petal.width"])
    0

def fix_species(df: pd.DataFrame) -> None:
    """Fixes species"""
    tmp = df[df["variety"] not in []]

    1


def main()->None:
    """"Main function"""
    df = get_data(f"{Path.cwd()}/assets/iris_with_errors.csv")
    print_empty_fields(df)
    fix_ranges(df)
    fix_species(df)

    df = pd.read_csv(f"{Path.cwd()}/assets/iris_with_errors.csv")

    # print(df.head())
    # print_empty_fields(df)
    fix_ranges(df)


if __name__ == "__main__":
    main()
