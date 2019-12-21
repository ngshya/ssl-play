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

            model_kmeans = KMeans(n_clusters=int(Xtot_scaled.shape[0]/30.0), random_state=1102, n_jobs=12)
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

        return self.model_rf.predict(X)
