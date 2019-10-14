import time
import numpy as np
import gc
import pandas as pd
from cvxopt import matrix, solvers
from functools import reduce
import math
from synthetic_data import sythetic_data
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import zero_one_loss
from tqdm import tqdm
from tqdm._tqdm import trange
import sys

plt.figure()
np.set_printoptions(suppress=True)


class JieyiSVM(object):
    def __init__(self,
                 data_path,
                 sampleNo,
                 mu1=None,
                 mu2=None,
                 sigma1=None,
                 sigma2=None,
                 add_noise=False,
                 noise_ratio=0.0):

        self.data_path = data_path
        self.sampleNo = sampleNo
        if add_noise :
            self.m = 2*sampleNo + int(2*sampleNo*noise_ratio)
        else:
            self.m = 2*self.sampleNo
        self.mu1 = mu1
        self.mu2 = mu2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.add_noise = add_noise
        self.noise_ratio = noise_ratio
        self.X = None
        self.Y = None
        self.testX = None
        self.testY = None
        self.quadratic_matrix = None

    def read_data(self, mode):
        """

        :return:
        """

        if self.data_path == "synthetic":
            X, Y = sythetic_data(self.mu1, self.mu2, self.sigma1, self.sigma2, self.sampleNo, self.add_noise, self.noise_ratio)
        else:
            data = pd.read_table(self.data_path, delimiter=",",header=None,names=['feature1', 'feature2', 'feature3', 'feature4', 'label'])
            data['label'].loc[(data['label'] == 'Iris-setosa')] = 1
            data['label'].loc[(data['label'] == 'Iris-versicolor')] = 0
            data = data.sample(frac=1)
            X = data[['feature1', 'feature2', 'feature3', 'feature4']].values
            Y = data[['label']].values

        if mode == "train":
            self.X = X
            self.Y = Y
        elif mode == "test":
            self.testX = X
            self.testY = Y
        return X, Y

    @staticmethod
    def gaussian_kernel(x, z, sigma):

        k = math.exp(reduce(lambda a, b: a**2+b**2, x-z)/(-2*sigma**2))
        return k

    def basic_info(self, C):
        """
        L(t) = p(t^2 + t) (t>=0)
             = 0 (t<0)
        :return:
        """
        # X, Y = self.read_data()
        Y_reshape = np.reshape(self.Y, [1, self.m])  # 1*m
        # print(np.shape(Y_reshape))
        yiyj = np.dot(np.transpose(Y_reshape), Y_reshape)  # m*m

        xixj = np.dot(self.X, np.transpose(self.X))  # m*m

        yiyjxixj = np.multiply(yiyj, xixj)

        yiyjxixj = np.multiply(1/2, yiyjxixj)

        # P1 = np.multiply(1/2,np.add(yiyjxixj, 1/(2*C)))

        tmp = np.array([1/(4*C)]*self.m)
        tmp_diag = np.diag(tmp)

        P1 = np.add(yiyjxixj, tmp_diag)

        tmp = np.array([1/(2*C)]*self.m)
        P2 = np.diag(tmp)

        P3 = np.zeros([self.m, self.m])
        # P3 = P2[:,:]
        tmp = np.array([1/(4*C)]*self.m)
        P4 = np.diag(tmp)

        del tmp

        gc.collect()

        _P = np.concatenate((P1, P2), axis=1)
        __P = np.concatenate((P3, P4), axis=1)

        P = np.concatenate((_P, __P), axis=0)

        del _P
        del __P
        gc.collect()

        q = np.reshape([-3/2]*self.m + [-1/2]*self.m, [2*self.m, 1])

        a1 = np.reshape(np.concatenate((Y_reshape[0], np.zeros(self.m))), [1, 2*self.m])

        a2 = np.concatenate((np.negative(np.eye(self.m)), np.zeros([self.m, self.m])), axis=1)
        a3 = np.concatenate((np.zeros([self.m, self.m]), np.negative(np.eye(self.m))), axis=1)
        a4 = np.concatenate((np.negative(np.eye(self.m)), np.negative(np.eye(self.m))), axis=1)

        G = np.concatenate((a2, a3, a4), axis=0)
        h = np.reshape([0]*self.m + [0]*self.m + [-C]*self.m, [(3*self.m)])

        A = a1
        b = np.array([0])

        # 判断 P 是否是正定的

        try:
            R = cholesky(P)
        except Exception as e:
            print(e)
        quadratic_matrix = {"P": P,
                            "q": q,
                            "G": G,
                            "h": h,
                            "A": A,
                            "b": b
                            }
        self.quadratic_matrix = quadratic_matrix

        return quadratic_matrix

    def elasticSvm_basic_info(self, C1, C2):
        """

        L(t) = C1*t^2 + C2*|t|


        :param C1: least squares 's param
        :param C2: absolute 's param
        :return:
        """
        Y_reshape = np.reshape(self.Y, [1, self.m])  # 1*m
        # print(np.shape(Y_reshape))
        yiyj = np.dot(np.transpose(Y_reshape), Y_reshape)  # m*m

        xixj = np.dot(self.X, np.transpose(self.X))  # m*m

        yiyjxixj = np.multiply(yiyj, xixj)  # m*m
        # yiyjxixj = np.multiply(1 / 2, yiyjxixj)  # m*m

        tmp1 = np.array([1/(C1)]*self.m)
        tmp2 = np.array([1 / (2 * C1)] * self.m)

        tmp1_diag = np.diag(tmp1)
        tmp2_diag= np.diag(tmp2)

        P_beta_or_lambda = np.add(yiyjxixj, tmp2_diag)

        tmp_zero = np.zeros([self.m, self.m])
        P_beta_lambda = np.add(np.multiply(-2, yiyjxixj), tmp1_diag)

        P_row1 = np.concatenate([P_beta_or_lambda, P_beta_lambda], axis=1)
        # P_row2 = np.concatenate([tmp1_diag, tmp2_diag, tmp_zero], axis=1)
        P_row3 = np.concatenate([tmp_zero,  P_beta_or_lambda], axis=1)

        P = np.concatenate([P_row1, P_row3], axis=0) # (2m,2m)

        # add gc collect
        del P_row1, P_row3, P_beta_or_lambda, P_beta_lambda
        gc.collect()

        q = np.reshape([-C1/(2*C2)]*self.m + [-1-C1/(2*C2)]*self.m, [2*self.m, 1])

        A = np.reshape(np.concatenate([Y_reshape[0],-Y_reshape[0]]), (1, 2*self.m))

        b = np.array([0])

        tmp_eye = np.negative(np.eye(self.m))
        G1 = np.concatenate((tmp_eye, tmp_zero), axis=1)
        G2 = np.concatenate((tmp_zero, tmp_eye), axis=1)
        # G3 = np.concatenate((tmp_zero, tmp_zero, tmp_eye), axis=1)
        G4 = np.concatenate((tmp_eye, tmp_eye), axis=1)

        G = np.concatenate((G1, G2, G4), axis=0)

        del G1, G2, G4
        gc.collect()

        h = np.reshape([0]*2*self.m + [-C2]*self.m, [3*self.m, 1])

        try:
            R = cholesky(P)
            # eigvalue = np.linalg.eigvals(P)
            # print(eigvalue)
        except Exception as e:
            print(e)

        quadratic_matrix = {"P": P,
                            "q": q,
                            "G": G,
                            "h": h,
                            "A": A,
                            "b": b
                            }
        self.quadratic_matrix = quadratic_matrix

        return quadratic_matrix

    def qp_solver(self, quadratic_matrix):
        """
         minimize    (1/2)*x'*P*x + q'*x
         subject to  G*x <= h
                     A*x = b.
        :param C :
        """
        # quadratic_matrix = self.basic_info(C)
        P_ = matrix(quadratic_matrix['P'], tc='d')
        q_ = matrix(quadratic_matrix['q'], tc='d')
        G_ = matrix(quadratic_matrix['G'], tc='d')
        h_ = matrix(quadratic_matrix['h'], tc="d")
        A_ = matrix(quadratic_matrix["A"], tc='d')
        b_ = matrix(quadratic_matrix["b"], (1, 1), tc="d")

        sol = solvers.qp(P_, q_, G_, h_, A_, b_, options={'show_progress': False})
        self.sol = sol
        return sol

    def decision_func(self, sol, C, verbose=False):
        """

        :param sol: result of qp_solver, a dict
        :param C:
        :return:
        """
        self.jieyi_C = C
        alpha_beta = np.array(sol['x'])

        alpha = np.reshape(alpha_beta[:self.m, :], [1, self.m])
        y = np.reshape(self.Y, [1, self.m])

        alphaiyi = np.multiply(alpha, y)  # 1*m

        X_1 = self.X[:, 0]  # (self.m,)
        X_2 = self.X[:, 1]  # (self.m,)
        x_coe = np.dot(alphaiyi, X_1)  # const
        y_coe = np.dot(alphaiyi, X_2)  # const

        # X_self.mul_new_x = np.dot(X, new_x)
        #
        # left = np.self.multiply(alphaiyi, X_self.mul_new_x)

        tmp = np.dot(alphaiyi, np.dot(self.X, self.X.T))  # 1*m

        b = np.average(y - tmp)

        # method2: choose one data ,alpha!=0 (弃)
        # b = y[0][0] - np.dot(alphaiyi,np.dot(self.X,self.X[0]))[0]
        bias = -b / y_coe[0]
        w1 = -x_coe[0] / y_coe[0]
        if verbose :
            print("------------C = {0}-----------".format(str(C)))
            print("b: ", bias)
            print("w1: ", w1)
            print("w2: ", 1)

            print("{0}*X1 + ({1})*X2 + ({2}) = 0".format(str(x_coe[0]), str(y_coe[0]), str(b)))

            print("分类面：X2 ={0}*X1+{1}".format(str(-x_coe[0]/y_coe[0]), str(-b/y_coe[0])))

        plt.figure(1)
        x = np.linspace(-2, 2, 10)
        y = (-b - x_coe[0]*x) / y_coe[0]
        label = "c={}".format(str(C))
        plt.plot(x, y, label=label)

        self.jieyisvm_w1 = w1
        self.jieyisvm_b = bias

        return w1, bias

    def elastic_decision_func(self, sol, C1, C2, verbose=False):
        self.elas_C1 = C1
        self.elas_C2 = C2

        gamma_kesi = np.array(sol["x"])

        gamma = np.reshape(gamma_kesi[:self.m, :], [self.m, 1])

        kesi = np.reshape(gamma_kesi[self.m:, :], [self.m, 1])

        e = (kesi - C2) / (2 * C1)

        y = np.reshape(self.Y, [self.m, 1])

        b1 = -np.divide(np.multiply(kesi, e), gamma) + 1

        b1 = np.multiply(b1, y)  # m*1

        xixj = np.dot(self.X, self.X.T)  # m*m

        gammaiyi = np.multiply(gamma, y)  # m*1

        b2 = np.dot(xixj, gammaiyi)

        b = np.average(b1 - b2)

        X_1 = self.X[:, 0]  # (m,)
        X_2 = self.X[:, 1]  # (m,)

        x_coe = np.dot(np.reshape(gammaiyi, [1, self.m]), X_1)
        y_coe = np.dot(np.reshape(gammaiyi, [1, self.m]), X_2)

        bias = -b / y_coe[0]

        w1 = -x_coe[0] / y_coe[0]

        if verbose:
            print("------------C1 = {0}, C2 = {1}-----------".format(str(C1), str(C2)))
            print("b: ", bias)
            print("w1: ", w1)
            print("w2: ", 1)

            print("{0}*X1 + ({1})*X2 + ({2}) = 0".format(str(x_coe[0]), str(y_coe[0]), str(b)))

            print("分类面：X2 ={0}*X1+{1}".format(str(-x_coe[0] / y_coe[0]), str(-b / y_coe[0])))
        plt.figure(1)

        x = np.linspace(-2, 2, 10)

        y = (-b - x_coe[0] * x) / y_coe[0]

        label = "c1 = {0}, c2 = {1}".format(str(C1), str(C2))
        plt.plot(x, y, label=label)

        self.elas_w1 = w1
        self.elas_b = bias
        return w1, bias


    def base_svm(self, C=1.,verbose=False):

        self.csvm_C = C

        clf = svm.SVC(kernel='linear', gamma=0.8, decision_function_shape='ovr', C=C)

        clf.fit(self.X, self.Y)

        w = clf.coef_[0]
        w1 = -w[0] / w[1]
        b = -clf.intercept_[0]/w[1]
        xx = np.linspace(-2, 2, 10)
        yy = w1 * xx - (clf.intercept_[0]) / w[1]

        if verbose :
            print("----------base--------")
            print("b: ", b)
            print("w1: ", w1)



        # plot the parallels to the separating hyperplane that pass through the
        # support vectors
        # b = clf.support_vectors_[0]
        # yy_down = a * xx + (b[1] - a * b[0])
        # b = clf.support_vectors_[-1]
        # yy_up = a * xx + (b[1] - a * b[0])

        plt.figure(1)
        plt.plot(xx, yy, label="base")

        self.csvm_w1 = w1
        self.csvm_b = b
        return w1, b

    @staticmethod
    def bayes_clf():

        plt.figure(1)
        x = np.linspace(-2, 2, 10)
        y = 2.5*x

        plt.plot(x, y, label="bayes clf")
    @staticmethod
    def sign(x):
        x = np.array(x)
        x_ = np.where(x >= 0, 1, -1)
        return x_



    def clf_accuracy(self, mode):
        bias = 0
        w1 =0

        if mode == "jieyi":
            w1, bias = self.jieyisvm_w1, self.jieyisvm_b
        elif mode == "elastic":
            w1, bias = self.elas_w1, self.elas_b
        elif mode == "base":
            w1, bias = self.csvm_w1, self.csvm_b

        w = np.array([w1, 1])
        y = np.add(np.dot(self.testX, w), bias)
        label_pred = self.sign(y)

        loss = zero_one_loss(self.testY, label_pred)
        accuracy = 1- loss
        return accuracy






if __name__ == "__main__":
    # path = 'iris.data'
    path = 'synthetic'
    sampleNo = 100

    mu1, mu2 = [0.5, -3], [-0.5, 3]
    sigma1 = sigma2 = [[0.2, 0], [0, 3]]
    add_noise = True
    noise_ratio = 0.1
    jieyi_C = 0.0001
    base_C = 0.003

    svm_train = JieyiSVM(data_path=path, sampleNo=sampleNo, mu1=mu1, mu2=mu2, sigma1=sigma1, sigma2=sigma2, add_noise=add_noise, noise_ratio=noise_ratio)
    jieyi_w1, jieyi_b = [], []
    csvm_w1, csvm_b = [], []
    jieyi_loss=[]
    csvm_loss =[]

    for i in range(100):
        # data
        X, Y = svm_train.read_data(mode = "train")
        testX, testY = svm_train.read_data(mode="test")

        # train
        quadratic_matrix = svm_train.basic_info(C = jieyi_C)
        sol = svm_train.qp_solver(quadratic_matrix)
        w1, bias = svm_train.decision_func(sol, C=jieyi_C, verbose=False)
        jieyi_w1.append(w1)
        jieyi_b.append(bias)

        w1_base, bias_base = svm_train.base_svm(C=base_C, verbose=False)
        csvm_w1.append(w1_base)
        csvm_b.append(bias_base)
        time.sleep(0.01)

        # test
        jieyiloss = svm_train.clf_accuracy(mode="jieyi")
        csvmloss = svm_train.clf_accuracy(mode="base")
        jieyi_loss.append(jieyiloss)
        csvm_loss.append(csvmloss)

        print('complete percent : %10.8s%s' % (str(1*i/100*100), "%"), end='\r')


    jieyi_w1_mean, jieyi_w1_std = np.mean(jieyi_w1), np.std(jieyi_w1)
    jieyi_b_mean, jieyi_b_std = np.mean(jieyi_b), np.std(jieyi_b)

    csvm_w1_mean, csvm_w1_std = np.mean(csvm_w1), np.std(csvm_w1)
    csvm_b_mean, csvm_b_std = np.mean(csvm_b), np.std(csvm_b)

    jieyi_loss_mean = np.mean(jieyi_loss)
    csvm_loss_mean = np.mean(csvm_loss)

    print("jieyi_w1 :",'{0}+{1}'.format(str(jieyi_w1_mean), str(jieyi_w1_std)))
    print("jieyi_b :",'{0}+{1}'.format(str(jieyi_b_mean), str(jieyi_b_std)))
    print("csvm_w1 :",'{0}+{1}'.format(str(csvm_w1_mean), str(csvm_w1_std)))
    print("csvm_b :",'{0}+{1}'.format(str(csvm_b_mean), str(csvm_b_std)))

    print("\n", "-------------------------------------", "\n")
    # print(jieyi_loss)
    print("jieyi_loss: {}".format(str(jieyi_loss_mean)))
    print("csvm_loss: {}".format(str(csvm_loss_mean)))




























