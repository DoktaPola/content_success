import pandas as pd
from sklearn.model_selection import train_test_split

import src.constants as C
import src.schema as S


def split_df(df: pd.DataFrame,
             filter_release_year: int,
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

    test = sorted_df[sorted_df[S.RELEASE_YEAR] == filter_release_year]
    train = sorted_df[sorted_df[S.RELEASE_YEAR] != filter_release_year]

    test.drop([S.RELEASE_YEAR, S.ELEMENT_UID, S.NAME], axis=1, inplace=True)
    train.drop([S.RELEASE_YEAR, S.ELEMENT_UID, S.NAME], axis=1, inplace=True)

    X_test, y_test = test.drop(S.TARGET_RATING, axis=1), test[S.TARGET_RATING]
    X, y = train.drop(S.TARGET_RATING, axis=1), train[S.TARGET_RATING]
    X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                    test_size=test_size,
                                                    shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test
