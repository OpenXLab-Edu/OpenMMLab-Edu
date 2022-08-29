from sklearn.datasets import load_wine
from ..BaseEdu import *
import numpy as np



def knn_demo():
    model = KNNClassifier(n_neighbors=10)
    model.load_dataset(dataset=load_wine())
    model.train()
    test_data = np.array([[11.8, 4.39, 2.39, 29, 82, 2.86, 3.53, 0.21, 2.85, 2.8, 0.75, 3.78, 490]])
    model.inference(test_data)


def pca_demo():
    data = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    model = PCA(n_components='mle')
    model.load_dataset(dataset=data)
    model.train()
    model.inference(data)


if __name__ == '__main__':
    knn_demo()
    # pca_demo()
