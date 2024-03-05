import pandas as pd
from pathlib import Path



def main():
    df = pd.read_csv(f"{Path.cwd()}/assets/iris_with_errors.csv")
    print(df.head())
    pass

if __name__ == "__main__":
    main()