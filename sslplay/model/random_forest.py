from sklearn.ensemble import RandomForestClassifier
import numpy as np

class ModelRF:


    def __init__(self):
        np.random.seed(1102)
        self.model = RandomForestClassifier(max_depth=5, random_state=1102, n_jobs=6)
        self.name = "RANDOM-FOREST"


    def fit(self, X, y, Xu=None):
        np.random.seed(1102)
        self.model.fit(X, y)

    
    def predict(self, X):
        np.random.seed(1102)
        return self.model.predict_proba(X)
