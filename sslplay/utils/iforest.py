import numpy as np
import multiprocessing
import tensorflow as tf # to avoid core dumped error
from sklearn.ensemble import IsolationForest

def iforest(X, num_estimators=100, random_state=1102, contamination=0.05):

    clf = IsolationForest(
        n_estimators=num_estimators,
        n_jobs=int(np.max([multiprocessing.cpu_count()-2, 1])), 
        random_state=random_state, 
        max_features=np.max([int(0.3 * X.shape[1]), 1]), 
        contamination=contamination

    )
    array_inliers = np.array(clf.fit_predict(X)) > 0

    return array_inliers