import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn-whitegrid')

dtf_performance_1 = pd.read_csv(
    "outputs/2020-01-04-01-26-experiment-1-b.csv", 
    sep=";", 
    decimal="."
)

dtf_performance_1.MODEL = dtf_performance_1.MODEL.map({
    "RANDOM-FOREST": "RF", 
    "KMEANS-RF": "KRF", 
    "LABEL-SPREADING": "LS"})

dtf_performance_1 = dtf_performance_1\
.loc[:, ["MODEL", "PERC_U", "METRIC", "VALUE_MEAN"]]\
.groupby(["PERC_U", "MODEL", "METRIC"])\
.agg({"VALUE_MEAN": lambda x: x}).unstack()

print(dtf_performance_1.to_latex(float_format="{:0.4f}".format))

dtf_performance_1.columns = [y for x,y in dtf_performance_1.columns]
dtf_performance_1 = dtf_performance_1.reset_index(drop=False)\
.sort_values(["PERC_U"])


x = sorted(np.unique(dtf_performance_1.PERC_U))

for str_metric in ["ACCURACY", "AUC_MACRO", "AUC_WEIGHTED", "F1_MACRO", "F1_WEIGHTED"]:

    y = dtf_performance_1[str_metric]

    for m in ["RF", "KRF", "LS"]:
        plt.plot(x, y[dtf_performance_1.MODEL == m], label=m)
    plt.legend()
    plt.title("AVERAGE " + str_metric.replace("_", " "))
    plt.xlim(xmin=79, xmax=79.9)
    plt.ylim(ymin=np.min(y[dtf_performance_1.MODEL == "RF"]), ymax=1)
    plt.xlabel("PERCENTAGE OF UNLABELED DATA")
    plt.ylabel(str_metric.replace("_", " "))
    plt.show()