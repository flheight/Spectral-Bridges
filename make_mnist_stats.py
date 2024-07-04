import numpy as np
from sklearn.decomposition import PCA
from spectralbridges import SpectralBridges
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

def evaluate_clustering(X, y, algorithm, n_clusters, n_iterations):
    indices = np.random.choice(X.shape[0], 20000, replace=False)
    X, y = X[indices], y[indices]

    ari_scores = []
    nmi_scores = []

    for i in range(n_iterations):
        match algorithm:
            case "KM":
                model = KMeans(n_clusters=n_clusters, random_state=i)
                labels = model.fit_predict(X)
            case "EM":
                model = GaussianMixture(n_components=n_clusters, random_state=i)
                labels = model.fit_predict(X)
            case "WC":
                model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
                labels = model.fit_predict(X)
            case "SB":
                model = SpectralBridges(n_clusters=n_clusters, n_nodes=500, random_state=i)
                model.fit(X)
                labels = model.predict(X)
            case _:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

        ari_scores.append(adjusted_rand_score(y, labels))
        nmi_scores.append(normalized_mutual_info_score(y, labels))

        print(f"{algorithm} iteration : {i}")

        if algorithm == "WC":
            break

    avg_ari = np.mean(ari_scores)
    avg_nmi = np.mean(nmi_scores)
    
    return avg_ari, avg_nmi

np.random.seed(0)

# Load data
data = np.genfromtxt('datasets/mnist_train.csv', delimiter=',')[1:]
X, y = data[:, 1:], data[:, 0]
#X = PCA(n_components=64, random_state=42).fit_transform(X)

n_clusters = 10
n_iterations = 10
algorithms = ["KM", "EM", "WC", "SB"]

# Evaluate and print results
for algorithm in algorithms:
    avg_ari, avg_nmi = evaluate_clustering(X, y, algorithm, n_clusters, n_iterations)
    print(f"{algorithm} - Adjusted Rand Index: {avg_ari:.4f}, Normalized Mutual Information: {avg_nmi:.4f}")
