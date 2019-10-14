import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt


def sythetic_data(mu1, mu2, sigma1, sigma2, sampleNo, add_noise=False, noise_ratio=0.0):
    """

    :param mu1:
    :param mu2:
    :param sigma1:
    :param sigma2:
    :param sampleNo:
    :return:
    """
    # np.random.seed(2019)
    # 生成两个二维的正态分布
    mu1 = np.array(mu1)
    sigma1 = np.array(sigma1)
    R = cholesky(sigma1)
    s = np.dot(np.random.randn(sampleNo, 2), R) + mu1
    y = np.ones([sampleNo,1]) # label = 1
    y = np.negative(np.ones([sampleNo, 1]))

    data1 = np.concatenate((s,y),axis=1)

    mu2 = np.array(mu2)
    sigma2 = np.array(sigma2)
    R2 = cholesky(sigma2)
    # np.random.seed(2019)
    s2 = np.dot(np.random.randn(sampleNo, 2), R2) + mu2
    y2 = np.ones([sampleNo, 1]) # label == -1
    data2 = np.concatenate((s2, y2), axis=1)
    plt.figure(1)

    plt.plot(s[:, 0], s[:, 1], '+')
    plt.plot(s2[:, 0], s2[:, 1], '+', color='red')
    # plt.show()

    data = np.concatenate((data1, data2), axis= 0)
    if add_noise == True:
        noise_sampleNo = int(2* sampleNo * noise_ratio)
        noise_mu = np.array([0,0])
        noise_sigma = np.array([[1, -0.8], [-0.8, 1]])
        noise_R = cholesky(noise_sigma)
        # np.random.seed(2019)
        noise_s = np.dot(np.random.randn(noise_sampleNo, 2), noise_R) + noise_mu
        noise_y = np.random.choice([-1, 1], size=noise_sampleNo).reshape(noise_sampleNo,1)
        noise_data = np.concatenate((noise_s, noise_y), axis=1)
        data = np.concatenate((data, noise_data), axis=0)

    np.random.shuffle(data)

    X = data[:,:-1]
    Y = data[:,-1]

    return X, Y



if __name__ == "__main__":

    data = sythetic_data()
