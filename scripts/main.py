#%%
import sys
import os
sys.path.append('.') 


#%%
#import importlib
#importlib.reload(sslplay)


#%%
from sslplay.data.spambase import DataSpambase


#%%
obj_data_spambase = DataSpambase()
obj_data_spambase.load()
obj_data_spambase.parse()
dict_split_out = obj_data_spambase.split()


# %%
