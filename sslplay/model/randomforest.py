from sklearn.ensemble import RandomForestClassifier

class modelRF:


    def __init__(self):
        self.model = RandomForestClassifier(max_depth=5, random_state=1102, n_jobs=16)


    def fit(self, X, y):
        self.model.fit(X, y)

    
    def predict(self, X):
        return self.model.predict_proba(X)
