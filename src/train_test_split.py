import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import src.constants as C
import src.schema as S


def split_df(df: pd.DataFrame,
             split_filter_year: int,
             target_name: str,
             test_size=None):
    """
    Split dataset on train/test by user_id.
    Parameters
    ----------
    X (pd.DataFrame): Initial dataset
    test_size (float or int, default=None): If float, should be between 0.0 and 1.0 and represent the proportion
                                            of the dataset to include in the test split. If int, represents the
                                            absolute number of test samples. If None, the value will
                                            be set to 0.1.
    Returns
    -------
    train and test datasets (pd.DataFrame)
    """
    if not test_size:
        test_size = C.DEFAULT_TEST_SIZE

    sorted_df = df.sort_values(S.RELEASE_YEAR)
    print(f'split_filter_year: {split_filter_year}')
    test = sorted_df[sorted_df[S.RELEASE_YEAR] == split_filter_year]
    train = sorted_df[sorted_df[S.RELEASE_YEAR] != split_filter_year]

    # test.drop([S.ELEMENT_UID, S.NAME], axis=1, inplace=True)
    # train.drop([S.ELEMENT_UID, S.NAME], axis=1, inplace=True)
    test.drop([S.RELEASE_YEAR, S.ELEMENT_UID, S.NAME], axis=1, inplace=True)
    train.drop([S.RELEASE_YEAR, S.ELEMENT_UID, S.NAME], axis=1, inplace=True)

    X_test, y_test = test.drop(target_name, axis=1), test[target_name]
    X, y = train.drop(target_name, axis=1), train[target_name]
    if target_name == 'WORLD':
        y = y.apply(lambda x: np.log1p(x))
        y_test = y_test.apply(lambda x: np.log1p(x))
    X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                      test_size=test_size,
                                                      shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test
