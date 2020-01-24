import sys
import os
sys.path.append('.')

import os

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

import pandas as pd
import numpy as np
import time

from sslplay.data.digits import DataDigits
from sslplay.model.random_forest import ModelRF

from sslplay.utils.ssplit import ssplit

from sslplay.performance.f1 import f1
from sslplay.performance.auc import auc
from sslplay.performance.accuracy import accuracy



obj_data = DataDigits()
obj_data.load()
obj_data.parse()




Xt, yt, Xl, yl = ssplit(
    obj_data.X, obj_data.y, 
    20, 80,
)

tmp_len = len(yl)

for p in [80, 60, 40, 20, 5, 2, 1]:
    q = p/80.0
    array_bool = np.random.choice(
        a=[False, True], 
        size=tmp_len, 
        p=[1-q, q], 
        replace=True
    )
    Xtrain = Xl[array_bool, :]
    ytrain = yl[array_bool]
    obj_model = ModelRF()
    obj_model.fit(Xtrain, ytrain, None)
    array_test_pred = obj_model.predict(Xt)
    array_test_real = yt
    logging.info(str(p) + " --> " \
    + str(accuracy(array_test_real, array_test_pred)))

"""
2020-01-23 23:35:49,135 - 80 --> 0.8866071428571428
2020-01-23 23:35:49,386 - 60 --> 0.89375
2020-01-23 23:35:49,631 - 40 --> 0.9196428571428571
2020-01-23 23:35:49,878 - 20 --> 0.8964285714285715
2020-01-23 23:35:50,126 - 5 --> 0.8848214285714285
2020-01-23 23:35:50,377 - 2 --> 0.7446428571428572
2020-01-23 23:35:50,626 - 1 --> 0.5928571428571429
"""