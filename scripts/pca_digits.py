import sys
sys.path.append('.')
from sslplay.data.digits import DataDigits

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

obj_data = DataDigits()
obj_data.load()

X = obj_data.X
y = obj_data.y
target_names = np.array(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
colors = np.array(['black', 'lime', 'darkorange', 
'darkred', 'chocolate', 'yellow', 'olive', 
'cyan', 'darkgrey', 'darkgreen'])

scaler = StandardScaler()
pca = PCA(n_components=6)
np.random.seed(1102)

X_r = pca.fit(scaler.fit_transform(X)).transform(scaler.fit_transform(X))
array_subset = np.random.choice(range(X_r.shape[0]), size=500, replace=False)
X_r = X_r[array_subset, :]
y_r = y[array_subset]


array_classes_show = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
plt.figure()
for color, i, target_name in zip(colors[array_classes_show], \
array_classes_show, target_names[array_classes_show]):
    plt.scatter(
        X_r[y_r == i, 0], X_r[y_r == i, 1], 
        color=color, alpha=.8, lw=2,   
        label=target_name
    )
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.suptitle('PCA - DIGITS')
plt.title("Explained variance ratio: %s"
      % str(pca.explained_variance_ratio_[[0, 1]]))
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend(fancybox=True, framealpha=1)
plt.show()

array_classes_show = [1,2,3,6,8,9]
plt.figure()
for color, i, target_name in zip(colors[array_classes_show], \
array_classes_show, target_names[array_classes_show]):
    plt.scatter(
        X_r[y_r == i, 2], X_r[y_r == i, 3], 
        color=color, alpha=.8, lw=2,   
        label=target_name
    )
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.suptitle('PCA - DIGITS')
plt.title("Explained variance ratio: %s"
      % str(pca.explained_variance_ratio_[[2, 3]]))
plt.xlabel("Dimension 3")
plt.ylabel("Dimension 4")
plt.legend(fancybox=True, framealpha=1)
plt.show()

array_classes_show = [2,6,8,9]
plt.figure()
for color, i, target_name in zip(colors[array_classes_show], \
array_classes_show, target_names[array_classes_show]):
    plt.scatter(
        X_r[y_r == i, 4], X_r[y_r == i, 5], 
        color=color, alpha=.8, lw=2,   
        label=target_name
    )
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.suptitle('PCA - DIGITS')
plt.title("Explained variance ratio: %s"
      % str(pca.explained_variance_ratio_[[4, 5]]))
plt.xlabel("Dimension 5")
plt.ylabel("Dimension 6")
plt.legend(fancybox=True, framealpha=1)
plt.show()

