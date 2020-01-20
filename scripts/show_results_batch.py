import pandas as pd
import pickle 
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
plt.style.use('seaborn-whitegrid')


str_run_id = time.strftime("%Y-%m-%d-%H-%M")

dtf_performance = pd.DataFrame()

for file in glob.glob('outputs/batch_run_output/*.pickle'):
    dtf_performance = dtf_performance.append(pickle.load(open(file, "rb")))

#dtf_performance.to_csv("outputs/" + str_run_id + "-batch.csv", sep=";", decimal=".", index=False)

dtf_performance.MODEL = dtf_performance.MODEL.map({
    "RANDOM-FOREST": "RF", 
    "NEURAL-NETWORK": "NN", 
    "KMEANS-RF": "KRF", 
    "LADDER-NETWORK": "LN", 
    "LABEL-SPREADING": "LS"})

dtf_performance_orig = dtf_performance.copy()

for str_dataset in np.unique(dtf_performance.DATASET):

    dtf_performance = dtf_performance_orig\
    .loc[dtf_performance_orig.DATASET == str_dataset, ["MODEL", "PERC_U", "METRIC",     "VALUE_MEAN"]]\
    .groupby(["PERC_U", "MODEL", "METRIC"])\
    .agg({"VALUE_MEAN": np.mean}).unstack()
    
    #print(dtf_performance.to_latex(float_format="{:0.4f}".format))
    
    dtf_performance.columns = [y for x,y in dtf_performance.columns]
    dtf_performance = dtf_performance.reset_index(drop=False)\
    .sort_values(["PERC_U"])
    
    
    x = sorted(np.unique(dtf_performance.PERC_U))
    
    for str_metric in ["ACCURACY", "AUC_MACRO", "AUC_WEIGHTED", "F1_MACRO", "F1_WEIGHTED"]:
    
        y = dtf_performance[str_metric]
    
        for m in ["RF", "KRF", "NN", "LN", "LS"]:
            plt.plot(x, y[dtf_performance.MODEL == m], label=m)
        plt.legend()
        plt.title(str_dataset + " - AVERAGE " + str_metric.replace("_", " "))
        plt.xlim(xmin=40, xmax=75)
        #plt.ylim(ymin=np.min(y[dtf_performance.MODEL == "NN"])-0.01, ymax=1)
        plt.xlabel("PERCENTAGE OF UNLABELED DATA")
        plt.ylabel(str_metric.replace("_", " "))
        plt.savefig(Path("img/" + str_dataset + "_" + str_metric + ".png"), dpi=600)
        #plt.show()
        plt.clf()
