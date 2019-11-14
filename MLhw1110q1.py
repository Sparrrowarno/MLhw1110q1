import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn import cluster, mixture


class image:
    def __init__(self, img):
        length = img.shape[0]
        width = img.shape[1]
        original_shape = img.shape
        size = img.shape[0] * img.shape[1]
        img = img / 255  # normalizing
        self.img = img
        feature_vector = np.zeros((size, 5))
        for i in range(length):
            feature_vector[0 + i * width:(i + 1) * width, 0] = np.arange(0, 1, 1 / width)
            feature_vector[0 + i * width:(i + 1) * width, 1] = (i / length) * np.ones(width)
        feature_vector[:, 2] = np.reshape(img[:, :, 0], (size, 1)).squeeze()
        feature_vector[:, 3] = np.reshape(img[:, :, 1], (size, 1)).squeeze()
        feature_vector[:, 4] = np.reshape(img[:, :, 2], (size, 1)).squeeze()
        self.feature_vector = feature_vector

    def k_mean_clustering(self, k):
        k_means = cluster.KMeans(n_clusters=k, max_iter=9999, n_init=4)
        k_means.fit(self.feature_vector)
        C_table = np.array([[255, 0, 0],
                            [0, 255, 0],
                            [0, 0, 255],
                            [255, 255, 0],
                            [0, 255, 255]])
        labels = k_means.labels_
        r = np.choose(labels, C_table[0:k, 0])
        g = np.choose(labels, C_table[0:k, 1])
        b = np.choose(labels, C_table[0:k, 2])
        r.shape = self.img[:, :, 0].shape
        g.shape = self.img[:, :, 0].shape
        b.shape = self.img[:, :, 0].shape
        color = np.dstack((r, g, b))
        return color

    def GMM_clustering(self, k):
        GMM = mixture.GaussianMixture(n_components=k, n_init=4)
        GMM.fit(self.feature_vector)
        C_table = np.array([[255, 0, 0],
                            [0, 255, 0],
                            [0, 0, 255],
                            [255, 255, 0],
                            [0, 255, 255]])
        labels = GMM.predict(self.feature_vector)
        r = np.choose(labels, C_table[0:k, 0])
        g = np.choose(labels, C_table[0:k, 1])
        b = np.choose(labels, C_table[0:k, 2])
        r.shape = self.img[:, :, 0].shape
        g.shape = self.img[:, :, 0].shape
        b.shape = self.img[:, :, 0].shape
        color = np.dstack((r, g, b))
        return color

    bird = cv2.imread('./img/EECE5644_2019Fall_Homework4Questions_42049_colorBird.jpg')
    plane = cv2.imread('./img/EECE5644_2019Fall_Homework4Questions_3096_colorPlane.jpg')

    img_bird = image(bird)
    img_plane = image(plane)

    plt.figure(figsize=(10, 10))
    for k in range(2, 6):
        plt.subplot(2, 2, k - 1)
        img_output = img_bird.k_mean_clustering(k)
        plt.imshow(img_output)
        plt.title('K-means: K = %d' % k)

    plt.figure(figsize=(10, 10))
    for k in range(2, 6):
        plt.subplot(2, 2, k - 1)
        img_output = img_bird.GMM_clustering(k)
        plt.imshow(img_output)
        plt.title('GMM: K = %d' % k)

    plt.figure(figsize=(10, 10))
    for k in range(2, 6):
        plt.subplot(2, 2, k - 1)
        img_output = img_plane.k_mean_clustering(k)
        plt.imshow(img_output)
        plt.title('K-means: K = %d' % k)

    plt.figure(figsize=(10, 10))
    for k in range(2, 6):
        plt.subplot(2, 2, k - 1)
        img_output = img_plane.GMM_clustering(k)
        plt.imshow(img_output)
        plt.title('GMM: K = %d' % k)

    plt.show()