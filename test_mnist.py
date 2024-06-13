import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from spectralbridges import SpectralBridges

from sklearn.metrics import adjusted_rand_score

np.random.seed(0)

data = np.genfromtxt('datasets/mnist_train.csv', delimiter=',')[1:]

X, y = data[:, 1:], data[:, 0]

X = PCA(n_components=64, random_state=42).fit_transform(X)

k = 10

net = SpectralBridges(n_clusters=k, n_nodes=500)

ari = 0
for i in range(10):
    net.fit(X)
    guess = net.predict(X)
    ari += adjusted_rand_score(y, guess) / 10

print(f"Adjusted Rand Index: {ari}")
