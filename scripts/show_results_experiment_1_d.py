import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn-whitegrid')

dtf_performance = pd.read_csv(
    "outputs/2020-01-24-13-50-experiment-1-d.csv", 
    sep=";", 
    decimal="."
)

dtf_performance.MODEL = dtf_performance.MODEL.map({
    "RANDOM-FOREST": "RF", 
    "KMEANS-RF": "KRF", 
    "NEURAL-NETWORK": "NN",
    "LADDER-NETWORK": "LN",
    "LABEL-SPREADING": "LS"})

dtf_performance = dtf_performance\
.loc[:, ["PERC_UNLA", "MODEL", "METRIC", "VALUE"]]\
.groupby(["PERC_UNLA", "MODEL", "METRIC"])\
.agg({"VALUE": np.mean}).unstack()

#print(dtf_performance.to_latex(float_format="{:0.4f}".format))

dtf_performance.columns = [y for x,y in dtf_performance.columns]
dtf_performance = dtf_performance.reset_index(drop=False)\
.sort_values(["PERC_UNLA"])


x = sorted(np.unique(dtf_performance.PERC_UNLA))

for str_metric in ["ACCURACY", "AUC_MACRO", "AUC_WEIGHTED", "F1_MACRO", "F1_WEIGHTED"]:

    y = dtf_performance[str_metric]

    for m in ["RF", "KRF", "NN", "LN"]:
        y_tmp = y[dtf_performance.MODEL == m]
        plt.plot(x, y_tmp, label=m)
    plt.legend()
    plt.title("AVERAGE " + str_metric.replace("_", " "))
    plt.xlim(xmin=5, xmax=40)
    plt.ylim(ymax=1)
    plt.xlabel("PERCENTAGE OF UNLABELED DATA")
    plt.ylabel(str_metric.replace("_", " "))
    plt.show()