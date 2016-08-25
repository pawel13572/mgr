import numpy as np
from sklearn.decomposition import PCA
import help_functions
import matplotlib.pyplot as plt


X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

training_set= help_functions.generate_set_ALL("C:/Users/admin/PycharmProjects/Forex/training_set.csv")
vector=[]
for i in range(0,len(training_set[0])):
    vector.append([training_set[0][i],training_set[1][i],training_set[2][i],training_set[3][i],training_set[4][i],
                  training_set[5][i], training_set[6][i], training_set[7][i], training_set[8][i], training_set[9][i]])

pca = PCA(n_components=None)
zz=pca.fit_transform(vector)
#print(pca.explained_variance_ratio_)

plt.plot(zz)
plt.show()