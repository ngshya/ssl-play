import sys
import os
sys.path.append('.') 

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

import numpy as np
import pandas as pd

from sslplay.model.random_forest import ModelRF
from sslplay.data.spambase import DataSpambase

#obj_model = ModelRF()
obj_data = DataSpambase()
obj_data.load()
obj_data.parse()
Xl, yl, Xu, yu, Xt, yt = obj_data.split(percentage_test=20, percentage_unlabelled=70, percentage_labelled=10)

#y[0] = 3
#obj_model.fit(X, y, None)
#array_test_pred = obj_model.predict(dict_split_out["test"])