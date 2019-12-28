import sys
sys.path.append('.')
from sslplay.data.spambase import DataSpambase

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

obj_data = DataSpambase()
obj_data.load()

X = obj_data.X
y = obj_data.y
target_names = ["Ham", "Spam"]

scaler = StandardScaler()
pca = PCA(n_components=2)
np.random.seed(1102)

X_r = pca.fit(X).transform(scaler.fit_transform(X))
array_subset = np.random.choice(range(X_r.shape[0]), size=300, replace=False)
X_r = X_r[array_subset, :]
y_r = y[array_subset]

print('explained variance ratio (first two components): %s' \
% str(pca.explained_variance_ratio_))

plt.figure()
colors = ['turquoise', 'darkorange']

for color, i, target_name in zip(colors, [0, 1], target_names):
    plt.scatter(
        X_r[y_r == i, 0], X_r[y_r == i, 1], 
        color=color, alpha=.8, lw=1,   
        label=target_name
    )
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.suptitle('PCA - SPAMBASE')
plt.title("explained variance ratio: %s" \
% str(pca.explained_variance_ratio_))
plt.xlim(-288.6, -288)
plt.ylim(-5.6, -5)
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()