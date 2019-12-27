from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sslplay.model.random_forest import ModelRF
from sslplay.utils.kmean_n_clusters import get_optimal_n_cluster
import numpy as np
import logging
import multiprocessing

class ModelKMeansRF:


    def __init__(self):
        np.random.seed(1102)
        self.model = RandomForestClassifier(
            max_depth=5, random_state=1102, 
            n_jobs=np.max([multiprocessing.cpu_count()-2, 1])
        )

        self.name = "KMEANS-RF"


    def fit(self, X, y, Xu):

        np.random.seed(1102)

        Xtot = np.vstack((X, Xu))
        array_bool_labelled = np.append(np.repeat(True, X.shape[0]), np.repeat(False, Xu.shape[0]))
        ytot = np.append(np.array(y), np.repeat(-1, Xu.shape[0]))

        if Xu.shape[0] > 0:

            scaler = MinMaxScaler()
            Xtot_scaled = scaler.fit_transform(Xtot)

            #int_opt_n_clusters = get_optimal_n_cluster(Xtot_scaled)

            #int_opt_n_clusters = np.min([int_opt_n_clusters, X.shape[0]])

            #logging.debug("Optimal number of clusters: " + str(int_opt_n_clusters))

            #initial_centers = X[np.random.choice(range(X.shape[0]), size=int_opt_n_clusters, replace=False), :]

            model_kmeans = KMeans(n_clusters=int(Xtot.shape[0] / 30.0), random_state=1102, n_jobs=6)
            model_kmeans.fit(Xtot_scaled)
            labels_kmeans = np.array(model_kmeans.labels_)

            for k in np.unique(sorted(labels_kmeans)):
                obj_model = ModelRF()
                array_bool_l_tmp = array_bool_labelled & (labels_kmeans == k)
                array_bool_u_tmp = (~array_bool_labelled) & (labels_kmeans == k)

                if (sum(array_bool_l_tmp) > 0) and (sum(array_bool_u_tmp) > 0):

                    X_tmp = Xtot[array_bool_l_tmp, :]
                    y_tmp = ytot[array_bool_l_tmp]

                    if len(np.unique(y_tmp)) == 1:
                        ytot[array_bool_u_tmp] = y_tmp[0]
                    else:
                        obj_model.fit(X_tmp, y_tmp)
                        tmp_y_values = sorted(np.unique(y_tmp))
                        ytot[array_bool_u_tmp] = np.take(tmp_y_values, obj_model.predict(Xtot[array_bool_u_tmp, :]).argmax(axis=1))

        Xtot = Xtot[ytot >= 0, :]
        ytot = ytot[ytot >=0]

        obj_model = ModelRF()
        obj_model.fit(Xtot, ytot)

        self.model_rf = obj_model

    
    def predict(self, X):
        np.random.seed(1102)
        return self.model_rf.predict(X)
