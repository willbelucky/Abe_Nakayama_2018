# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018-09-21
"""
import pandas as pd
from ksif import Portfolio
from ksif.core.columns import *
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

if __name__ == '__main__':
    columns = [DATE, CODE, RET_1, B_P, E_P, DIVP, S_P, C_P, ROE, ROA, ROIC, S_A, LIQ_RATIO, EQUITY_RATIO, ASSETSYOY,
               BETA_3M, MKTCAP, MOM1, MOM12, VOL_3M, TRADING_VOLUME_RATIO]

    rolling_columns = [B_P, E_P, DIVP, S_P, C_P, ROE, ROA, ROIC, S_A, LIQ_RATIO, EQUITY_RATIO, ASSETSYOY,
                       BETA_3M, MKTCAP, MOM1, MOM12, VOL_3M, TRADING_VOLUME_RATIO]

    pf = Portfolio()
    pf = pf.loc[~pd.isna(pf[RET_1]), :]
    months = sorted(pf[DATE].unique())

    result_columns = [DATE, CODE, RET_1]
    rolled_columns = []
    all_set = pf.reset_index(drop=True)
    for column in rolling_columns:
        t_0 = column + '_t'
        t_1 = column + '_t-1'
        t_2 = column + '_t-2'
        t_3 = column + '_t-3'
        t_4 = column + '_t-4'
        t_5 = column + '_t-5'
        t_6 = column + '_t-6'
        t_7 = column + '_t-7'
        t_8 = column + '_t-8'
        t_9 = column + '_t-9'
        t_10 = column + '_t-10'
        t_11 = column + '_t-11'
        t_12 = column + '_t-12'
        result_columns.extend([t_0, t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8, t_9, t_10, t_11, t_12])
        rolled_columns.extend([t_0, t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8, t_9, t_10, t_11, t_12])
        all_set[t_0] = all_set[column]
        all_set[t_1] = all_set.groupby(by=[CODE]).apply(lambda x: x[column].shift(1)).reset_index(drop=True)
        all_set[t_2] = all_set.groupby(by=[CODE]).apply(lambda x: x[column].shift(2)).reset_index(drop=True)
        all_set[t_3] = all_set.groupby(by=[CODE]).apply(lambda x: x[column].shift(3)).reset_index(drop=True)
        all_set[t_4] = all_set.groupby(by=[CODE]).apply(lambda x: x[column].shift(4)).reset_index(drop=True)
        all_set[t_5] = all_set.groupby(by=[CODE]).apply(lambda x: x[column].shift(5)).reset_index(drop=True)
        all_set[t_6] = all_set.groupby(by=[CODE]).apply(lambda x: x[column].shift(6)).reset_index(drop=True)
        all_set[t_7] = all_set.groupby(by=[CODE]).apply(lambda x: x[column].shift(7)).reset_index(drop=True)
        all_set[t_8] = all_set.groupby(by=[CODE]).apply(lambda x: x[column].shift(8)).reset_index(drop=True)
        all_set[t_9] = all_set.groupby(by=[CODE]).apply(lambda x: x[column].shift(9)).reset_index(drop=True)
        all_set[t_10] = all_set.groupby(by=[CODE]).apply(lambda x: x[column].shift(10)).reset_index(drop=True)
        all_set[t_11] = all_set.groupby(by=[CODE]).apply(lambda x: x[column].shift(11)).reset_index(drop=True)
        all_set[t_12] = all_set.groupby(by=[CODE]).apply(lambda x: x[column].shift(12)).reset_index(drop=True)

    all_set = all_set[result_columns]
    all_set = all_set.dropna().reset_index(drop=True)
    all_set[rolled_columns] = scaler.fit_transform(all_set[rolled_columns])
    all_set.to_csv('data/all_set.csv', index=False)
