import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt



edge_file = open("case1.edge",'w')

adj = np.zeros((900, 900))
for i in range(300):
    for j in range(i+1, 300):
        z = np.random.randint(0, 100, dtype=int)
        if z > 96:  #0.03
            adj[i, j] = 1
            adj[j, i] = 1
            
for i in range(300, 600):
    for j in range(i+1, 600):
        z = np.random.randint(0, 100, dtype=int)
        if z > 96: #0.03
            adj[i, j] = 1
            adj[j, i] = 1
            
for i in range(600, 900):
    for j in range(i+1, 900):
        z = np.random.randint(0, 100, dtype=int)
        if z > 96: #0.03
            adj[i, j] = 1
            adj[j, i] = 1
            
for i in range(300):
    for j in range(300, 600):
        z = np.random.randint(0, 100, dtype=int)
        if z > 96: #0.03
            adj[i, j] = 1
            adj[j, i] = 1
            
for i in range(300):
    for j in range(600, 900):
        z = np.random.randint(0, 100, dtype=int)
        if z > 96: #0.03
            adj[i, j] = 1
            adj[j, i] = 1

for i in range(300, 600):
    for j in range(600, 900):
        z = np.random.randint(0, 100, dtype=int)
        if z > 96: #0.03
            adj[i, j] = 1
            adj[j, i] = 1

x, y = np.where(adj > 0)
for i in range(len(x)):
    if x[i] != y[i]:
       edge_file.write('{} {}\n'.format(x[i], y[i]))
edge_file.close()


dim = 50
mask_convariance_maxtix = np.diag([1 for i in range(dim)])

center1 = 2.5 * np.random.random(size=dim) - 1
center2 = 2.5 * np.random.random(size=dim) - 1
center3 = 2.5 * np.random.random(size=dim) - 1

center = np.vstack((center1, center2, center3))

data1 = multivariate_normal.rvs(mean=center1, cov=mask_convariance_maxtix, size=300)
data2 = multivariate_normal.rvs(mean=center2, cov=mask_convariance_maxtix, size=300)
data3 = multivariate_normal.rvs(mean=center3, cov=mask_convariance_maxtix, size=300)
data = np.vstack((data1, data2, data3))


label = np.array([0 for i in range(300)] + [1 for i in range(300)] + [2 for i in range(300)])

permutation = np.random.permutation(label.shape[0])

data = data[permutation, :]
label = label[permutation]

np.savetxt('case1.feature', data, fmt='%f')
np.savetxt('case1.label', label, fmt='%d')

