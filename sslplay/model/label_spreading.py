import numpy as np
from sklearn.semi_supervised import LabelSpreading
from sklearn.preprocessing import MinMaxScaler
import multiprocessing

class ModelLabelSpreading:


    def __init__(self):
        np.random.seed(1102)
        self.model = LabelSpreading(
            kernel="rbf", 
            n_jobs=np.max([multiprocessing.cpu_count()-2, 1]), 
            alpha=0.2, n_neighbors=10, max_iter=15
        )
        self.name = "LABEL-SPREADING"
        self.scaler = MinMaxScaler()


    def fit(self, X, y, Xu=None):
        np.random.seed(1102)
        self.Xl = self.scaler.fit_transform(X)
        self.yl = y
        #self.Xu = Xu


    def predict(self, X):
        np.random.seed(1102)
        self.Xt = self.scaler.transform(X)
        X = np.vstack((self.Xl, self.Xt))
        y = np.append(self.yl, np.repeat(-1, self.Xt.shape[0]))
        #y = np.append(y, np.repeat(-1, self.Xt.shape[0]))
        y = np.int64(y)

        assert X.shape[0] == len(y)

        self.model.fit(X, y)

        return np.array(self.model.label_distributions_)[(-self.Xt.shape[0]):, :]
