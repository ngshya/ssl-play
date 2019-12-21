#%%
import sys
import os
sys.path.append('.') 


import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING)


#%%
from sslplay.model.test_models import test_models
from sslplay.model.kmeans_random_forest import ModelKMeansRF
from sslplay.model.random_forest import ModelRF

dtf_performance = test_models(array_perc_unla=[79.9, 79.5, 79, 75], array_models=[ModelRF, ModelKMeansRF])


#%%
dtf_performance.to_csv("out.csv", sep=";", index=False)