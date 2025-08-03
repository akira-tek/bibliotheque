""" A file that contains a bunch of useful utilities for machine learning related topics.

Current contents are:
- `DataFrameVectorizer` (vectorize values from given columns much like CountVectorizer from scikit-learn)
"""

# dependencies #
import numpy as np
import pandas as pd

#     utils    #
class DataFrameVectorizer:
  def __init__(self, columns: list | np.ndarray | pd.Index):
    """ Initializes a `DataFrameVectorizer` instance.
    Parameters:
      columns (list, np.ndarray, pd.Index): an array of columns to vectorize.
    """
    self.columns: list = columns
    self.column2categories: dict[str, str] = {}
    self.is_train_fit: bool = False

  def fit_train(self, df: pd.DataFrame) -> pd.DataFrame:
    """ Vectorizes the train dataframe.
    Parameters:
      df (pd.DataFrame): dataframe to vectorize.
    Returns:
      pd.DataFrame: dataframe vectorized by columns given at initialization.
    """
    dataframe = df.copy()
    for column in self.columns:
      dataframe[column] = dataframe[column].astype("category")
      self.column2categories[column] = dataframe[column].cat.categories
      dataframe[column] = dataframe[column].cat.codes

    self.is_train_fit = True
    return dataframe

  def fit_test(
      self,
      df: pd.DataFrame,
      fill_unknown: int | float | None = None
  ) -> pd.DataFrame:
    """ Vectorized the test dataframe. `fit_train` must be used before this.
    Parameters:
      df (pd.DataFrame): dataframe to vectorize.
      fill_unknown (int, float, optional): Values to use instead of codes
        for categories not found in the train dataframe but found in the
        test dataframe. To use the maximum value (i. e. length of the
        column2categories dictionary generated after calling `fit_train`),
        don't specify anything (default: `None`).
    
    Returns:
      pd.DataFrame: dataframe vectorized by columns given at initialization.
    """
    assert self.is_train_fit, "The train dataframe must be fit before the test dataframe."

    value_to_replace_unknown = len(self.column2categories) if fill_unknown is None else fill_unknown
    dataframe = df.copy()
    for column in self.columns:
      dataframe[column] = pd.Categorical(dataframe[column], categories=self.column2categories[column])
      dataframe[column] = dataframe[column].cat.codes
      dataframe[column] = dataframe[column].replace(-1, value_to_replace_unknown)
      
    return dataframe
