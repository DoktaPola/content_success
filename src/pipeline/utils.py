import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import style
from sklearn.preprocessing import StandardScaler

style.use('seaborn')



def standardize_df(sc_cols: list, X_train,
                   X_val, X_test):
    sc = StandardScaler()

    # train
    sc_train = sc.fit_transform(X_train[sc_cols])
    sc_train = pd.DataFrame(sc_train, index=X_train.index, columns=sc_cols)
    X_train.drop(sc_cols, axis=1, inplace=True)
    X_train = pd.concat([X_train, sc_train],axis=1)

    # val
    sc_val = sc.transform(X_val[sc_cols])
    sc_val = pd.DataFrame(sc_val, index=X_val.index, columns=sc_cols)
    X_val.drop(sc_cols, axis=1, inplace=True)
    X_val = pd.concat([X_val, sc_val],axis=1)

    # test
    sc_test = sc.transform(X_test[sc_cols])
    sc_test = pd.DataFrame(sc_test, index=X_test.index, columns=sc_cols)
    X_test.drop(sc_cols, axis=1, inplace=True)
    X_test = pd.concat([X_test, sc_test],axis=1)

    return X_train, X_val, X_test


def target_distr_linear(train_label, test_label, predicted_dv):
    sns.kdeplot(train_label, label='train')
    sns.kdeplot(test_label, label='test')
    sns.kdeplot(predicted_dv, label='pred')
    plt.legend()
    plt.show()

    # ddd = pd.DataFrame({'test_lbl': test_label, 'pred_lbl':predicted_dv})
    sns.scatterplot(test_label, predicted_dv, color='blueviolet')
    plt.title('Linear model')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.show()