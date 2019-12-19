#%%
import sys
import os
sys.path.append('.') 


#%%
from sslplay.model.test_models import test_models
from sslplay.model.kmeans_random_forest import ModelKMeansRF
dtf_performance = test_models()


#%%
dtf_performance.to_csv("out.csv", sep=";")