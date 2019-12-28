'''
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import logging

def calculate_wcss(data):
    wcss = []
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    for n in range(2, 21):
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(X=data)
        logging.debug("KMeans " + str(n) + " clusters WCSS: " + str(round(kmeans.inertia_, 4)))
        wcss.append(kmeans.inertia_)
    return wcss


def optimal_number_of_clusters(wcss):
    x1, y1 = 2, wcss[0]
    x2, y2 = 20, wcss[len(wcss)-1]

    distances = []
    for i in range(len(wcss)):
        x0 = i+2
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)
    
    return distances.index(max(distances)) + 2


def get_optimal_n_cluster(data):
    return optimal_number_of_clusters(calculate_wcss(data))
'''