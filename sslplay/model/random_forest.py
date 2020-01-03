from sklearn.ensemble import RandomForestClassifier
import numpy as np
import multiprocessing

class ModelRF:


    def __init__(self):
        np.random.seed(1102)
        self.model = RandomForestClassifier(
            n_estimators=25,
            max_depth=4, 
            random_state=1102, 
            n_jobs=int(np.max([multiprocessing.cpu_count()-2, 1]))
        )
        self.name = "RANDOM-FOREST"


    def fit(self, X, y, Xu=None):
        np.random.seed(1102)
        self.model.fit(X, y)

    
    def predict(self, X):
        np.random.seed(1102)
        return self.model.predict_proba(X)
