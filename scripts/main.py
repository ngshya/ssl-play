#%%
import sys
import os
sys.path.append('.') 


#%%
#import importlib
#importlib.reload(sslplay)


#%%
from sslplay.data.spambase import DataSpambase
from sslplay.model.randomforest import modelRF
from sslplay.performance.auc import auc
from sslplay.performance.f1 import f1


#%%
obj_data_spambase = DataSpambase()
obj_data_spambase.load()
obj_data_spambase.parse()
obj_model_rf = modelRF()

for port_unla in [0.0, 0.2, 0.5, 0.8, 0.9]:
    dict_split_out = obj_data_spambase.split(port_test=0.2, port_unla=port_unla)
    obj_model_rf.fit(dict_split_out["train_l"], dict_split_out["target_train_l"])
    array_test_pred = obj_model_rf.predict(dict_split_out["test"])
    array_test_real = dict_split_out["target_test"]
    print("AUC: " + str(auc(array_test_real, array_test_pred)))
    print("F1: " + str(f1(array_test_real, array_test_pred)))

# %%
