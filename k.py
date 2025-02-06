import numpy as np
import matplotlib.pyplot as plt

# 生成模拟的西瓜数据集4.0
def generate_watermelon_data():
    np.random.seed(0)
    X, y = np.array([
        [0.697, 0.460], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318], [0.556, 0.215], 
        [0.403, 0.237], [0.481, 0.149], [0.437, 0.211], [0.666, 0.091], [0.243, 0.267], 
        [0.245, 0.057], [0.343, 0.099], [0.639, 0.161], [0.657, 0.198], [0.360, 0.370], 
        [0.593, 0.042], [0.719, 0.103]
    ]), None
    return X

# 实现K均值算法
def kmeans(X, k, max_iter=100, init_centers=None):
    if init_centers is not None:
        centers = np.array(init_centers)
    else:
        np.random.seed(42)
        centers = X[np.random.choice(X.shape[0], k, replace=False)]
    
    for _ in range(max_iter):
        distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centers = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        if np.all(centers == new_centers):
            break
        centers = new_centers
    
    return centers, labels

# 实验配置与运行
def run_experiments():
    data = generate_watermelon_data()
    k_values = [2, 3, 4]
    init_centers_list = [
        [[0.4, 0.2], [0.8, 0.6]],
        [[0.6, 0.2], [0.4, 0.4], [0.7, 0.3]],
        [[0.6, 0.1], [0.4, 0.3], [0.8, 0.6], [0.5, 0.5]]
    ]

    results = []
    for k, init_centers in zip(k_values, init_centers_list):
        centers, labels = kmeans(data, k, init_centers=init_centers[:k])
        results.append((k, centers, labels))

    return data, results

# 结果可视化
def plot_all_results(data, results):
    fig, axes = plt.subplots(1, len(results), figsize=(15, 5))
    for ax, (k, centers, labels) in zip(axes, results):
        ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o', label='Data points')
        ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=100, label='Centers')
        ax.set_title(f'k-means clustering (k={k})')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.legend()
    plt.tight_layout()
    plt.show()

# 主程序运行
if __name__ == "__main__":
    data, results = run_experiments()
    plot_all_results(data, results)

