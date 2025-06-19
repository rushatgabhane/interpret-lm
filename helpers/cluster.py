from sklearn.cluster import KMeans
import numpy as np


class ExplanationClusterer:
    def __init__(self, n_clusters=10):
        self.kmeans = KMeans(n_clusters=n_clusters)

    def fit(self, explanation_vectors):
        self.kmeans.fit(explanation_vectors)

    def predict(self, vector):
        return self.kmeans.predict([vector])[0]

    def cluster_centers(self):
        return self.kmeans.cluster_centers_
