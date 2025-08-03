""" A file that contains a bunch of useful utilities for machine learning related topics.

Current contents are:
- `DataFrameVectorizer` (vectorize values from given columns much like CountVectorizer from scikit-learn)
"""

# dependencies #
import numpy as np
import pandas as pd

#     utils    #
class DataFrameVectorizer:
  def __init__(self,
               columns: list | np.ndarray | pd.Index):
    self.columns: list = columns
    self.column2categories: dict[str, str] = {}
    self.is_train_fit: bool = False

  def fit_train(self,
                df: pd.DataFrame):
    dataframe = df.copy()
    for column in self.columns:
      dataframe[column] = dataframe[column].astype("category")
      self.column2categories[column] = dataframe[column].cat.categories
      dataframe[column] = dataframe[column].cat.codes

    self.is_train_fit = True
    return dataframe

  def fit_test(self,
               df: pd.DataFrame):
    assert self.is_train_fit, "The train dataframe must be fit before the test dataframe."

    dataframe = df.copy()
    for column in self.columns:
      dataframe[column] = pd.Categorical(dataframe[column], categories=self.column2categories[column])
      dataframe[column] = dataframe[column].cat.codes
      # replacing every not found value (-1) as the length of the dictionary (in this case - max(dict.values()) + 1)
      # since -1 is used to represent nan values
      dataframe[column] = dataframe[column].replace(-1, len(self.column2categories))

    return dataframe