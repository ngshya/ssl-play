#%%
import sys
import os
sys.path.append('.') 


#%%
from sslplay.data.spambase import DataSpambase
from sslplay.data.creditcard import DataCreditCard
from sslplay.model.randomforest import modelRF
from sslplay.performance.auc import auc
from sslplay.performance.f1 import f1


#%%
obj_data = DataSpambase()
obj_data.load()
obj_data.parse()
obj_model = modelRF()

for port_unla in [0.0, 0.5, 0.9, 0.99, 0.995]:
    dict_split_out = obj_data.split(port_test=0.2, port_unla=port_unla)
    obj_model.fit(dict_split_out["train_l"], dict_split_out["target_train_l"])
    array_test_pred = obj_model.predict(dict_split_out["test"])
    array_test_real = dict_split_out["target_test"]
    print("Unlabelled data portion: " + str(port_unla))
    print("AUC: " + str(auc(array_test_real, array_test_pred)))
    print("F1: " + str(f1(array_test_real, array_test_pred)))
    print("")


# %%
obj_data = DataCreditCard()
obj_data.load()
obj_data.parse()
obj_model = modelRF()

for port_unla in [0.0, 0.5, 0.9, 0.99, 0.995]:
    dict_split_out = obj_data.split(port_test=0.2, port_unla=port_unla)
    obj_model.fit(dict_split_out["train_l"], dict_split_out["target_train_l"])
    array_test_pred = obj_model.predict(dict_split_out["test"])
    array_test_real = dict_split_out["target_test"]
    print("Unlabelled data portion: " + str(port_unla))
    print("AUC: " + str(auc(array_test_real, array_test_pred)))
    print("F1: " + str(f1(array_test_real, array_test_pred)))
    print("")