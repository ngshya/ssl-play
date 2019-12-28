import sys
sys.path.append('.')
from sslplay.data.digits import DataDigits

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

obj_data = DataDigits()
obj_data.load()

X = obj_data.X
y = obj_data.y
target_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

scaler = StandardScaler()
pca = PCA(n_components=3)
np.random.seed(1102)

X_r = pca.fit(X).transform(scaler.fit_transform(X))
array_subset = np.random.choice(range(X_r.shape[0]), size=500, replace=False)
X_r = X_r[array_subset, :]
y_r = y[array_subset]

fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(X_r[:, 0], X_r[:, 1], X_r[:, 2], c=y_r,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()