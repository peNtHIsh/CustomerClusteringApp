from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from minisom import MiniSom
import numpy as np

class Clustering:
    def kmeans_clustering(self, scaled_data, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        return clusters

    def mini_batch_kmeans_clustering(self, scaled_data, n_clusters, batch_size=100):
        mbk = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=42)
        clusters = mbk.fit_predict(scaled_data)
        return clusters

    def dbscan_clustering(self, scaled_data, eps=0.5, min_samples=5):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(scaled_data)
        return clusters

    def som_clustering(self, scaled_data, som_size=10, iterations=100):
        som = MiniSom(som_size, som_size, scaled_data.shape[1], sigma=1.0, learning_rate=0.5)
        som.random_weights_init(scaled_data)
        som.train_random(scaled_data, iterations)
        positions = []
        for x in scaled_data:
            w = som.winner(x)
            positions.append(w)
        positions = np.array(positions)
        return positions  # Возвращаем позиции BMU для каждого образца

    def calculate_elbow_method(self, scaled_data, use_mini_batch=False):
        inertia = []
        silhouette_scores = []
        K = range(2, 11)

        for k in K:
            if use_mini_batch:
                kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=100)
            else:
                kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(scaled_data)
            inertia.append(kmeans.inertia_)
            silhouette_avg = silhouette_score(scaled_data, labels)
            silhouette_scores.append(silhouette_avg)
        return K, inertia, silhouette_scores

    def calculate_silhouette(self, scaled_data, clusters):
        score = silhouette_score(scaled_data, clusters)
        return score