# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018-09-21
"""
import os
from keras.activations import relu
from keras.initializers import Zeros, lecun_normal
from tqdm import tqdm

from model import *

if __name__ == '__main__':

    base = os.path.basename(__file__)
    file_name = os.path.splitext(base)[0]

    test_pf = pf.loc[pf[DATE] >= '2012-05-31', :]
    test_months = sorted(test_pf[DATE].unique())

    MSE_list = []
    RMSE_list = []
    CORR_list = []
    long_short_tertile_return_list = []
    top_tertile_return_list = []
    bottom_tertile_return_list = []
    long_short_quintile_return_list = []
    top_quintile_return_list = []
    bottom_quintile_return_list = []

    param = {
        'batch_size': 300,
        'epochs': 100,
        'bias_initializer': Zeros(),
        'kernel_initializer': lecun_normal(),
        'activation': relu,
        'hidden_layers': [120, 120, 80, 80, 40, 40],
        'dropout': False,
        'dropout_rate': 0.5
    }

    for month in tqdm(test_months):
        model, X_train, actual_train, X_test, actual_test = train_model(month, param)

        MSE, RMSE, CORR, long_short_tertile_return, top_tertile_return, bottom_tertile_return, \
        long_short_quintile_return, top_quintile_return, bottom_quintile_return = get_results(model, X_test,
                                                                                              actual_test)

        MSE_list.append(MSE)
        RMSE_list.append(RMSE)
        CORR_list.append(CORR)
        long_short_tertile_return_list.append(long_short_tertile_return)
        top_tertile_return_list.append(top_tertile_return)
        bottom_tertile_return_list.append(bottom_tertile_return)
        long_short_quintile_return_list.append(long_short_quintile_return)
        top_quintile_return_list.append(top_quintile_return)
        bottom_quintile_return_list.append(bottom_quintile_return)

    df_result = pd.DataFrame(data={
        DATE: test_months,
        'MSE': MSE_list,
        'RMSE': RMSE_list,
        'CORR': CORR_list,
        'long_short_tertile_return': long_short_tertile_return_list,
        'top_tertile_return': top_tertile_return_list,
        'bottom_tertile_return': bottom_tertile_return_list,
        'long_short_quintile_return': long_short_quintile_return_list,
        'top_quintile_return': top_quintile_return_list,
        'bottom_quintile_return': bottom_quintile_return_list,
    })
    df_result.to_csv('data/{}.csv'.format(file_name), index=False)
