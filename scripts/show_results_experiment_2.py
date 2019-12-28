import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn-whitegrid')

dtf_performance_2 = pd.read_csv(
    "outputs/2019-12-28-12-14-experiment-2.csv", 
    sep=";", 
    decimal="."
)

dtf_performance_2.MODEL = dtf_performance_2.MODEL.map({
    "RANDOM-FOREST": "RF", 
    "NEURAL-NETWORK": "NN", 
    "KMEANS-RF": "KRF", 
    "LADDER-NETWORK": "LN", 
    "LABEL-SPREADING": "LS"})

dtf_performance_2 = dtf_performance_2\
.loc[:, ["MODEL", "DATASET", "METRIC", "VALUE_MEAN"]]\
.groupby(["DATASET", "MODEL", "METRIC"])\
.agg({"VALUE_MEAN": lambda x: x}).unstack()

print(dtf_performance_2.to_latex(float_format="{:0.4f}".format))

dtf_performance_2.columns = [y for x,y in dtf_performance_2.columns]
dtf_performance_2 = dtf_performance_2.reset_index(drop=False)\
.sort_values(["DATASET"])

colors = np.array(['b', 'g', 'r', 'c', 'm'])
datasets = np.unique(dtf_performance_2.DATASET)
x = np.arange(len(datasets)) / 2.2


for str_perf in ["ACCURACY", "AUC_MACRO", "AUC_WEIGHTED", "F1_MACRO", "F1_WEIGHTED"]:
    plt.figure(figsize=(8,5))
    for i, str_model in enumerate(["RF", "NN", "KRF", "LN", "LS"]):
        y = dtf_performance_2[str_perf][dtf_performance_2["MODEL"] == str_model]
        plt.bar(
            x+i/15, y, color = colors[i], 
            width = 0.05, label=str_model, alpha=0.8
        )
    plt.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, 1.15),
        ncol=3, fancybox=True, shadow=True
    )
    plt.xticks(x + 2/15, datasets)
    plt.ylabel("AVERAGE " + str_perf.replace("_", " "))
    plt.ylim((0, 1))
    plt.show()