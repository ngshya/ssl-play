from sklearn.ensemble import RandomForestClassifier

class ModelRF:


    def __init__(self):
        self.model = RandomForestClassifier(max_depth=5, random_state=1102, n_jobs=6)
        self.name = "RANDOM-FOREST"


    def fit(self, X, y, Xu=None):
        self.model.fit(X, y)

    
    def predict(self, X):
        return self.model.predict_proba(X)
