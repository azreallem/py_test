import numpy as np
import pandas as pd


# Basic data structures in pandas
# Series: a one-dimensional labeled array holding data of any type
# DataFrame: a two-dimensional data structure that holds data like a two-dimension array or a table with rows and columns.

# Object creation
def mytest01():
    s = pd.Series([1, 3, 5, np.nan, 6, 8])

    dates = pd.date_range("20130101", periods=6)
    df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list("ABCD"))

    df2 = pd.DataFrame(
        {
             "A": 1.0,
             "B": pd.Timestamp("20130102"),
             "C": pd.Series(1, index=list(range(4)), dtype="float32"),
             "D": np.array([3] * 4, dtype="int32"),
             "E": pd.Categorical(["test", "train", "test", "train"]),
             "F": "foo",
        }
    )
    print(df2)
    print(df2.dtypes)

# Viewing data
def mytest02():
    dates = pd.date_range("20130101", periods=6)
    df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list("ABCD"))
    df2 = pd.DataFrame(
        {
             "A": 1.0,
             "B": pd.Timestamp("20130102"),
             "C": pd.Series(1, index=list(range(4)), dtype="float32"),
             "D": np.array([3] * 4, dtype="int32"),
             "E": pd.Categorical(["test", "train", "test", "train"]),
             "F": "foo",
        }
    )

    print(df.head())
    print(df.tail(3))
    print(df.index)
    print(df.columns)
    print(df.to_numpy())
    print(df2)
    print(df2.dtypes)
    print(df2.to_numpy())
    print(df.describe())    # shows a quick statistic summary of your data
    print(df)
    print(df.T)
    


































































































#----------------------------------------main----------------------------------------#
def main():
    mytest02()

if __name__ == '__main__':
    main()
