print(__doc__)

import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs


def _generateData():
    """
    Two close clusters, one big and the other small,
    :return:
    """
    X = np.random.uniform(2, 8, size=(500, 300))
    y= np.zeros(len(X))
    # # Visualize the test data
    # fig0, ax0 = plt.subplots()
    # for label in range(3):
    #     ax0.plot(X[y == label][:, 0], X[y == label][:, 1], '.',
    #              color=colors[label])
    # ax0.set_xlim(0.1, 3)
    # ax0.set_ylim(-0.75, 2.75)
    # ax0.set_title('Test data: 200 points x3 clusters.')
    return X, y

np.random.seed(0)

batch_size = 45
centers = [[1, 1], [-1, -1], [1, -1]]
n_clusters = len(centers)
# X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7)
X, labels_true= _generateData()

k_means = KMeans(init='k-means++', n_clusters=10, n_init=10)
t0 = time.time()
k_means.fit(X)
t_batch = time.time() - t0

fig = plt.figure(figsize=(8, 6))
# fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
colors = ['#4EACC5', '#FF9C34', '#4E9A06']*30

# We want to have the same colors for the same cluster from the
# MiniBatchKMeans and the KMeans algorithm. Let's pair the cluster centers per
# closest one.
k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)
# KMeans
ax = fig.add_subplot(1,1,1)
for k, col in zip(range(10), colors):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=6)
ax.set_title('KMeans')
ax.set_xticks(())
ax.set_yticks(())
plt.text(-3.5, 1.8,  'train time: %.2fs\ninertia: %f' % (
    t_batch, k_means.inertia_))

plt.show()
