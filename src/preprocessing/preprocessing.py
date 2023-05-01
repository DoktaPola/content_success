from typing import Optional

import numpy as np
import pandas as pd

import src.schema as S
from src.core import BaseTransformer
from src.preprocessing.utils import add_topN_feature, ohe_topN_features


class SimplePreprocessor(BaseTransformer):
    """
    Transformer class for short and simple requests dataset preprocessing.
    Returns
    -------
    X : pd.DataFrame
        Preprocessed dataframe.
    """

    def __init__(self, target_name: str,
                 list_features: dict,
                 skewed_num_features: list,
                 filter_release_year: int):
        self.target_name = target_name
        self.filter_release_year = filter_release_year
        self.list_features = list_features
        self.skewed_num_features = skewed_num_features

    def _filter_year(
            self,
            df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Filter data by year.
        """
        df = df[df[S.RELEASE_YEAR] >= self.filter_release_year]
        return df

    def _add_top_n_features(
            self,
            df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Process list features and get N top.
        """
        df.dropna(subset=self.list_features.keys(), inplace=True)
        for key, val in self.list_features.items():
            topNfeature, name = add_topN_feature(df, key, val)
            df = ohe_topN_features(df, topNfeature, name)
        return df

    def _preprocess_skewed_num_features(
            self,
            df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Transform skewed features.
        """
        for col in self.skewed_num_features:
            df[col] = df[col].apply(lambda x: np.log1p(x))
        return df

    def _drop_undue_columns(
            self,
            df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Drop undue columns.
        """
        drop_list = [S.DIRECTOR, S.AVERAGE_RATING,
                     S.RELEASE_TYPE, S.ACTOR, S.ELEMENT_ID]
        if self.target_name == S.TARGET_WORLD:
            drop_list.append(S.TARGET_RATING)
        df.drop(drop_list,
                axis=1, inplace=True)
        return df

    def _transform_target(self, X: pd.DataFrame):
        X.dropna(subset=[S.TARGET_WORLD], inplace=True)
        return X

    def _fit_df(
            self,
            X: pd.DataFrame,
            y: Optional[pd.Series] = None
    ) -> None:
        pass

    def _transform_df(
            self,
            X: pd.DataFrame
    ) -> pd.DataFrame:
        if self.target_name == 'WORLD':
            X = self._transform_target(X)
        X = self._filter_year(X)
        X = self._add_top_n_features(X)
        X = self._preprocess_skewed_num_features(X)
        X = self._drop_undue_columns(X)
        return X
