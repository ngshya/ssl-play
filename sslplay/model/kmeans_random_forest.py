from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sslplay.model.random_forest import ModelRF
import numpy as np

class ModelKMeansRF:


    def __init__(self):
        self.model = RandomForestClassifier(max_depth=5, random_state=1102, n_jobs=16)

        self.name = "KMEANS-RF"


    def fit(self, X, y, Xu):

        Xtot = np.vstack((X, Xu))
        array_bool_labelled = np.append(np.repeat(True, X.shape[0]), np.repeat(False, Xu.shape[0]))
        ytot = np.append(np.array(y), np.repeat(-1, Xu.shape[0]))

        if Xu.shape[0] > 0:

            scaler = StandardScaler()
            Xtot_scaled = scaler.fit_transform(Xtot)

            model_kmeans = KMeans(n_clusters=20, random_state=1102)
            model_kmeans.fit(Xtot_scaled)
            labels_kmeans = np.array(model_kmeans.labels_)

            for k in np.unique(sorted(labels_kmeans)):
                obj_model = ModelRF()
                array_bool_tmp = array_bool_labelled & (labels_kmeans == k)
                if sum(array_bool_tmp) > 0:
                    obj_model.fit(Xtot[array_bool_tmp, :], ytot[array_bool_tmp])
                    tmp_y_values = sorted(np.unique(ytot[array_bool_tmp]))
                    array_bool_tmp = (~array_bool_labelled) & (labels_kmeans == k)
                    if sum(array_bool_tmp) > 0:
                        ytot[array_bool_tmp] = np.take(tmp_y_values, obj_model.predict(Xtot[array_bool_tmp, :]).argmax(axis=1))

        Xtot = Xtot[ytot >= 0, :]
        ytot = ytot[ytot >=0]

        obj_model = ModelRF()
        obj_model.fit(Xtot, ytot)

        self.model_rf = obj_model

    
    def predict(self, X):

        return self.model_rf.predict(X)
