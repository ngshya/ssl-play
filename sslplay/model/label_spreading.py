import numpy as np
from sklearn.semi_supervised import LabelSpreading

class ModelLabelSpreading:


    def __init__(self):
        self.model = LabelSpreading(kernel="knn", n_jobs=6, alpha=0.2, n_neighbors=10, max_iter=15)
        self.name = "LABEL-SPREADING"


    def fit(self, X, y, Xu=None):
        self.Xl = X
        self.yl = y
        #self.Xu = Xu


    def predict(self, X):
        self.Xt = X
        X = np.vstack((self.Xl, self.Xt))
        y = np.append(self.yl, np.repeat(-1, self.Xt.shape[0]))
        #y = np.append(y, np.repeat(-1, self.Xt.shape[0]))
        y = np.int64(y)

        assert X.shape[0] == len(y)

        self.model.fit(X, y)

        return np.array(self.model.label_distributions_)[(-self.Xt.shape[0]):, :]
