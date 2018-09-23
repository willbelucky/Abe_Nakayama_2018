# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018-09-23
"""
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential
from ksif import Portfolio
from ksif.core.columns import *
from scipy.stats import spearmanr

pf = Portfolio()
pf = pf.loc[~pd.isna(pf[RET_1]), :]
months = sorted(pf[DATE].unique())

result_columns = [RET_1]

all_set = pd.read_csv('data/all_set.csv', parse_dates=[DATE])


def get_data_set(test_month):
    test_index = months.index(test_month)
    assert test_index - 132 >= 0, "test_month is too early"

    train_start_month = months[test_index - 120]

    training_set = all_set.loc[(all_set[DATE] >= train_start_month) & (all_set[DATE] < test_month), :]
    test_set = all_set.loc[all_set[DATE] == test_month, :]

    return training_set, test_set


def train_model(month, param):
    tf.reset_default_graph()
    data_train, data_test = get_data_set(test_month=month)

    # Make data a numpy array
    data_train_array = data_train.values
    data_test_array = data_test.values

    X_train = data_train_array[:, 3:]
    y_train = data_train_array[:, 2:3]
    X_test = data_test_array[:, 3:]
    y_test = data_test_array[:, 2:3]
    actual_train = data_train.loc[:, [DATE, CODE, RET_1]].reset_index(drop=True)
    actual_test = data_test.loc[:, [DATE, CODE, RET_1]].reset_index(drop=True)

    input_dim = 234

    # Parameters
    batch_size = param['batch_size']
    epochs = param['epochs']
    bias_initializer = param['bias_initializer']
    kernel_initializer = param['kernel_initializer']
    activation = param['activation']
    hidden_layers = param['hidden_layers']
    dropout = param['dropout']
    dropout_rate = param['dropout_rate']

    model = Sequential()
    model.add(Dense(hidden_layers[0], input_dim=input_dim,
                    activation=activation,
                    bias_initializer=bias_initializer,
                    kernel_initializer=kernel_initializer))
    model.add(BatchNormalization())
    if dropout:
        model.add(Dropout(dropout_rate))

    for hidden_layer in hidden_layers[1:]:
        model.add(Dense(hidden_layer,
                        activation=activation,
                        bias_initializer=bias_initializer,
                        kernel_initializer=kernel_initializer
                        ))
        model.add(BatchNormalization())
        if dropout:
            model.add(Dropout(dropout_rate))

    model.add(Dense(1))
    model.compile(loss=keras.losses.mse,
                  optimizer=keras.optimizers.Adam())
    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=0,
              validation_data=(X_test, y_test))

    return model, X_train, actual_train, X_test, actual_test


def get_results(model, X, actual_y):
    predict_ret_1 = 'predict_' + RET_1
    actual_rank = 'actual_rank'
    predict_rank = 'predict_rank'

    prediction = model.predict(X, verbose=0)
    df_prediction = pd.concat([actual_y, pd.DataFrame(prediction, columns=[predict_ret_1])], axis=1)
    df_prediction['diff'] = df_prediction[RET_1] - df_prediction[predict_ret_1]
    df_prediction[actual_rank] = df_prediction[RET_1].rank()
    df_prediction[predict_rank] = df_prediction[predict_ret_1].rank()

    MSE = (df_prediction['diff'] ** 2).mean()
    RMSE = np.sqrt(MSE)

    CORR, _ = spearmanr(df_prediction[actual_rank], df_prediction[predict_rank])

    top_tertile_return = df_prediction.loc[df_prediction[predict_rank] > 0.6666 * len(df_prediction), RET_1].mean()
    bottom_tertile_return = df_prediction.loc[df_prediction[predict_rank] < 0.3333 * len(df_prediction), RET_1].mean()
    long_short_tertile_return = top_tertile_return - bottom_tertile_return

    top_quintile_return = df_prediction.loc[df_prediction[predict_rank] > 0.8 * len(df_prediction), RET_1].mean()
    bottom_quintile_return = df_prediction.loc[df_prediction[predict_rank] < 0.2 * len(df_prediction), RET_1].mean()
    long_short_quintile_return = top_quintile_return - bottom_quintile_return

    return MSE, RMSE, CORR, long_short_tertile_return, top_tertile_return, bottom_tertile_return, \
           long_short_quintile_return, top_quintile_return, bottom_quintile_return
