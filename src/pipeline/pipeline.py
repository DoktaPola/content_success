import logging
import os
import sys
from abc import ABC

import catboost as cat
import pandas as pd
import shap
from catboost import Pool

import src.schema as S
from src.core import BaseTransformer
from src.evaluate.metrics import calc_metrics
from src.pipeline.utils import standardize_df, target_distr_linear
from src.train_test_split import split_df

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s')
log = logging.getLogger('pipeline-log')
log.setLevel(logging.DEBUG)


class Pipeline(ABC):
    def __init__(self,
                 mode: str,
                 preprocessor: BaseTransformer,
                 model,
                 sandart_scaler_cols: list,
                 splitting_params: dict = None):
        """
        Pipeline to unite data processing, model training and predictions and counting scores for it's performance
        :param preprocessor: basic preprocessor instance
        :param augmenter: basic augmenter (synonyms imputer)
        :param convertor: custom convertor words to vectors
        :param model: custom model
        :param splitting_params: dict of parameters for splitting (see split_df)
        """
        self.mode = mode
        self.preprocessor = preprocessor
        self.model = model
        self.splitting_params = splitting_params
        self.sandart_scaler_cols = sandart_scaler_cols
        self.optimizer = None

    def run_ml_regressor(
            self,
            X: pd.DataFrame,
            cat_features: list,
            model_filename: str='model'
            ):
        """
        Run pipeline
        """
        df = X.copy()
        log.info('Running ML regressor pipeline')
        X_train, X_val, X_test, \
        y_train, y_val, y_test = self.prepare_data(df)

        train_dataset = cat.Pool(data=X_train,
                                label=y_train,
                                cat_features=cat_features)

        test_dataset = cat.Pool(data=X_test,
                                label=y_test,
                                cat_features=cat_features)
        val_dataset = cat.Pool(data=X_val,
                               label=y_val,
                               cat_features=cat_features)

        self.train_ml_model(train_dataset, val_dataset)
        y_pred = self.predict(test_dataset)
        self.draw_curves(y_train, y_test, y_pred)
        self.get_scores(y_test, y_pred)
        self.get_feature_importance(X_train, y_train,
                                    cat_features)
        self.save_model(filename=model_filename)


    # def run_regressor(
        #     self,
        #     X: pd.DataFrame,
        #     learning_rate,
        #     criterion: object,
        #     optimizer: object,
        #     epochs: int = 10,
        #     checkpoint_path='../checkpoints/best_checkpoint'
        #     ):
        # """
        # Run pipeline.
        # """
        # df = X.copy()
        # log.info('Running multi classifier pipeline')
        # train_df, val_df, test_df = self.prepare_data(df)
        # iteration_list, loss_list, accuracy_list = self.train_model(learning_rate,
        #                                                             criterion,
        #                                                             optimizer,
        #                                                             train_df,
        #                                                             val_df,
        #                                                             epochs=epochs,
        #                                                             checkpoint_path=checkpoint_path)

        # self.draw_curves(iteration_list, loss_list, accuracy_list)
        # pred = self.predict(test_df, checkpoint_path=checkpoint_path)
        # self.get_scores(test_df[S.TARGET].values, pred)

    def train_ml_model(self,
                       train_df: pd.DataFrame,
                       val_df: pd.DataFrame,
                       params:dict = None):
        """
        Train model
        :return: data for drawing curves
        """
        log.info('Training model')

        self.model.fit(train_df,
                       eval_set=val_df,
                       early_stopping_rounds=500,
                       verbose=100,
                       plot=False)

    # def train_dl_model(self,
    #                 learning_rate,
    #                 criterion: object,
    #                 optimizer: object,
    #                 train_df: pd.DataFrame,
    #                 val_df: pd.DataFrame,
    #                 batch_size=CONST.BATCH_SIZE,
    #                 epochs=CONST.EPOCHS,
    #                 iter_per_validation=100,
    #                 early_stopping=False,
    #                 checkpoint_path="./best_checkpoint",
    #                 device=CONF.DEVICE
    #                 ):
    #     """
    #     Train model
    #     :return: data for drawing curves
    #     """
    #     log.info('Training model')
    #     log.info(f'*** {self.model}')
        # model = self.model(n_tokens=len(self.convertor.get_tokens())).to(CONF.DEVICE)
        # optimizer = optimizer(model.parameters(), lr=learning_rate)

        # self.model = model
        # self.optimizer = optimizer

        # iteration_list, loss_list, accuracy_list = U.train(self.convertor, model,
        #                                                    optimizer, criterion,
        #                                                    train_df, val_df,
        #                                                    batch_size=batch_size,
        #                                                    epochs=epochs,
        #                                                    iter_per_validation=iter_per_validation,
        #                                                    early_stopping=early_stopping,
        #                                                    checkpoint_path=checkpoint_path,
        #                                                    device=device)

        # return iteration_list, loss_list, accuracy_list

    def draw_curves(self,
                    y_train=None,
                    y_test=None,
                    y_pred=None,
                    iteration_list=None,
                    loss_list=None,
                    accuracy_list=None):
        """
        Visualize train, test and pred of the ML model
        Visualize loss and accuracy of the DL model
        """
        if self.mode == 'ml':
            target_distr_linear(y_train.values, y_test.values, y_pred)
        elif self.mode == 'dl':
            pass
            # U.draw_visualization(iteration_list, loss_list, accuracy_list)

    def predict(self,
                X: pd.DataFrame,
                checkpoint_path='./best_checkpoint'
                ):
        """
        Get classes prediction.
        :return: classes
        """
        log.info('Starting prediction')
        if self.mode == 'ml':
            preds = self.model.predict(X)
        elif self.mode == 'dl':
            # _ = U.load_checkpoint(checkpoint_path, self.model, self.optimizer)

            # preds = []

            # with torch.no_grad():
            #     for batch in U.iterate_minibatches(self.convertor, X,
            #                                     batch_size=128, shuffle=False,
            #                                     device=CONF.DEVICE):
            #         test_outputs = self.model(batch)
            #         predicted = torch.max(test_outputs.data, 1)[1]
            #         preds.extend(predicted)
            pass

        return preds

    def get_scores(self, y_true, y_pred):
        calc_metrics(y_true, y_pred)

    def prepare_data(self, df: pd.DataFrame):
        """
        Prepare data for training and prediction
        :param df: Dataframe to prepare
        :return: three dataframes  X_train, X_val, X_test and targets for each.
        """
        df = df.copy()

        log.debug('Starting preprocessing')
        self.preprocessor.fit(df)
        df = self.preprocessor.transform(df)
        df.drop(columns=[S.POSTER_URL, S.POSTER_URL_PREVIEW],
                axis=1, inplace=True)

        log.debug('Splitting dataset on train, val and test')
        X_train, X_val, X_test, \
        y_train, y_val, y_test = split_df(df, **self.splitting_params)

        X_train, X_val, X_test = standardize_df(self.sandart_scaler_cols,
                                                X_train, X_val, X_test)
        return  X_train, X_val, X_test, \
                y_train, y_val, y_test

    def get_feature_importance(self, X_train, y_train,
                               cat_features):
        log.debug('Getting feature importance')
        shap.initjs()
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(Pool(X_train, y_train, cat_features=cat_features))
        shap.summary_plot(shap_values, X_train)

    def save_model(self, filename):
        log.debug('Saving model')
        if self.mode == 'ml':
            self.model.save_model('../src/models/'+filename, format="cbm")
        elif self.mode == 'dl':
            pass