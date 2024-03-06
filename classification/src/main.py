import pandas as pd
import numpy as np
from pathlib import Path

def get_empty_fields(df):
    print(df.isnull().sum())


def main():
    df = pd.read_csv(f"{Path.cwd()}/assets/iris_with_errors.csv")
    # print(df.head())
    get_empty_fields(df)
    pass

if __name__ == "__main__":
    main()