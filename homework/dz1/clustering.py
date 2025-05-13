import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


class DBSCANManual:
    def __init__(self, eps=0.8, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None
        self.n_clusters = 0

    def _get_neighbors(self, X, point_idx):
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        return np.where(distances <= self.eps)[0]

    def _expand_cluster(self, X, point_idx, neighbors, cluster_id):
        self.labels[point_idx] = cluster_id
        seeds = list(neighbors)

        i = 0
        while i < len(seeds):
            current_point = seeds[i]

            if self.labels[current_point] == -1:
                self.labels[current_point] = cluster_id
                current_neighbors = self._get_neighbors(X, current_point)

                if len(current_neighbors) >= self.min_samples:
                    for neighbor in current_neighbors:
                        if neighbor not in seeds:
                            seeds.append(neighbor)
            i += 1

    def fit(self, X):
        n_samples = X.shape[0]
        self.labels = np.full(n_samples, -1)
        cluster_id = 0

        for point_idx in range(n_samples):
            if self.labels[point_idx] != -1:
                continue

            neighbors = self._get_neighbors(X, point_idx)

            if len(neighbors) < self.min_samples:
                self.labels[point_idx] = -2
            else:
                self._expand_cluster(X, point_idx, neighbors, cluster_id)
                cluster_id += 1

        unique_labels = np.unique(self.labels)
        label_map = {}
        new_label = 0

        for label in unique_labels:
            if label == -2:
                label_map[label] = -1
            else:
                label_map[label] = new_label
                new_label += 1

        self.labels = np.array([label_map[label] for label in self.labels])
        self.n_clusters = new_label
        return self


def find_optimal_eps(X, k=5):
    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs.fit(X)
    distances, _ = nbrs.kneighbors(X)
    mean_distances = np.mean(distances[:, 1:], axis=1)
    return np.sort(mean_distances)


def main():
    df = pd.read_csv('data.csv')
    X = df.values
    print(f"Загружено {X.shape[0]} точек, {X.shape[1]} признаков")

    X_selected = X[:, [2, 6]]
    print(f"Используем признаки x3 и x7 (индексы 2 и 6)")

    sorted_distances = find_optimal_eps(X_selected, k=5)

    plt.figure(figsize=(10, 5))
    plt.plot(sorted_distances)
    plt.xlabel('Точки (отсортированные)')
    plt.ylabel('Среднее расстояние до 5 соседей')
    plt.title('График k-расстояний')
    plt.grid(True, alpha=0.3)

    elbow_point = len(sorted_distances) // 4
    eps = sorted_distances[elbow_point]
    plt.axhline(y=eps, color='r', linestyle='--', label=f'eps = {eps:.3f}')
    plt.legend()
    plt.show()

    min_samples = 4
    print(f"Параметры: eps={eps:.3f}, min_samples={min_samples}")

    dbscan = DBSCANManual(eps=eps, min_samples=min_samples)
    dbscan.fit(X_selected)
    labels = dbscan.labels
    n_clusters = dbscan.n_clusters
    n_noise = np.sum(labels == -1)

    print(f"Найдено кластеров: {n_clusters}")
    print(f"Шумовых точек: {n_noise} ({n_noise / len(X) * 100:.2f}%)")

    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        if label == -1:
            print(f"Шум: {count} точек")
        else:
            print(f"Кластер {label}: {count} точек")

    sample_size = min(5000, len(X_selected))
    np.random.seed(42)
    idx = np.random.choice(len(X_selected), sample_size, replace=False)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_selected[idx, 0], X_selected[idx, 1],
                          c=labels[idx], cmap='tab20', alpha=0.5, s=8)
    plt.colorbar(scatter, label='Кластер (-1 = шум)')
    plt.xlabel('x3')
    plt.ylabel('x7')
    plt.title(f'DBSCAN: {n_clusters} кластеров, eps={eps:.3f}')
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()