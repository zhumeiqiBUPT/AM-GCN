import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt



edge_file = open("case2.edge",'w')

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
        z = np.random.randint(0, 10000, dtype=int)
        if z > 9984: #0.0015
            adj[i, j] = 1
            adj[j, i] = 1
            
for i in range(300):
    for j in range(600, 900):
        z = np.random.randint(0, 10000, dtype=int)
        if z > 9984: #0.0015
            adj[i, j] = 1
            adj[j, i] = 1

for i in range(300, 600):
    for j in range(600, 900):
        z = np.random.randint(0, 10000, dtype=int)
        if z > 9984: #0.03
            adj[i, j] = 1
            adj[j, i] = 1

x, y = np.where(adj > 0)
for i in range(len(x)):
    if x[i] != y[i]:
       edge_file.write('{} {}\n'.format(x[i], y[i]))
edge_file.close()


dim = 50
mask_convariance_maxtix = np.diag([1 for i in range(dim)])

center = 2.5 * np.random.random(size=dim) - 1
data = multivariate_normal.rvs(mean=center, cov=mask_convariance_maxtix, size=900)

label = np.array([0 for i in range(300)] + [1 for i in range(300)] + [2 for i in range(300)])

np.savetxt('case2.feature', data, fmt='%f')
np.savetxt('case2.label', label, fmt='%d')

