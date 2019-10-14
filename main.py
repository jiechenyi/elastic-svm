from qp_solver import JieyiSVM
import numpy as np
import time



def cal_mean_std(x):
    return np.mean(x), np.std(x)

def simulation(data_path,
               sampleNo,
               mu1,
               mu2,
               sigma1,
               sigma2,
               add_noise,
               noise_ratio,
               jieyi_C,
               elas_C1,
               elas_C2,
               base_C,
               jieyi=True,
               elastic = True,
               base  = True

               ):
    svm_train = JieyiSVM(data_path=data_path, sampleNo=sampleNo, mu1=mu1, mu2=mu2, sigma1=sigma1, sigma2=sigma2, add_noise=add_noise, noise_ratio=noise_ratio)
    jieyi_w1, jieyi_b = [], []
    csvm_w1, csvm_b = [], []
    elas_w1, elas_b = [], []

    jieyi_loss=[]
    csvm_loss =[]
    elas_loss =[]

    for i in range(100):
        # data
        X, Y = svm_train.read_data(mode = "train")
        testX, testY = svm_train.read_data(mode="test")

        # train & test
        if jieyi is True:
            quadratic_matrix = svm_train.basic_info(C = jieyi_C)
            sol = svm_train.qp_solver(quadratic_matrix)
            w1, bias = svm_train.decision_func(sol, C=jieyi_C, verbose=False)
            jieyi_w1.append(w1)
            jieyi_b.append(bias)
            jieyiloss = svm_train.clf_accuracy(mode="jieyi")
            jieyi_loss.append(jieyiloss)

        # elastic train
        if elastic is True:
            qm = svm_train.elasticSvm_basic_info(elas_C1, elas_C2)
            elas_sol = svm_train.qp_solver(qm)
            w1, bias = svm_train.elastic_decision_func(elas_sol, elas_C1, elas_C2, verbose=False)
            elas_w1.append(w1)
            elas_b.append(bias)
            elasloss = svm_train.clf_accuracy(mode="elastic")
            elas_loss.append(elasloss)

        # base
        if base is True:
            w1_base, bias_base = svm_train.base_svm(C=base_C, verbose=False)
            csvm_w1.append(w1_base)
            csvm_b.append(bias_base)
            csvmloss = svm_train.clf_accuracy(mode="base")
            csvm_loss.append(csvmloss)

        time.sleep(0.01)

        # print('complete percent : %10.8s%s' % (str(1*i/100*100), "%"), end='\r')


    jieyi_w1_mean, jieyi_w1_std = cal_mean_std(jieyi_w1)
    jieyi_b_mean, jieyi_b_std = cal_mean_std(jieyi_b)

    csvm_w1_mean, csvm_w1_std = cal_mean_std(csvm_w1)
    csvm_b_mean, csvm_b_std = cal_mean_std(csvm_b)

    elas_w1_mean, elas_w1_std = cal_mean_std(elas_w1)
    elas_b_mean, elas_b_std = cal_mean_std(elas_b)


    jieyi_loss_mean = np.mean(jieyi_loss)
    csvm_loss_mean = np.mean(csvm_loss)
    elas_loss_mean = np.mean(elas_loss)

    print("jieyi_w1 :",'{0}+{1}'.format(str(jieyi_w1_mean), str(jieyi_w1_std)))
    print("jieyi_b :",'{0}+{1}'.format(str(jieyi_b_mean), str(jieyi_b_std)))
    print("csvm_w1 :",'{0}+{1}'.format(str(csvm_w1_mean), str(csvm_w1_std)))
    print("csvm_b :",'{0}+{1}'.format(str(csvm_b_mean), str(csvm_b_std)))
    print("elas_w1 :", '{0}+{1}'.format(str(elas_w1_mean), str(elas_w1_std)))
    print("elas_b :", '{0}+{1}'.format(str(elas_b_mean), str(elas_b_std)))

    print("\n", "-------------------------------------", "\n")
    # print(jieyi_loss)
    print("jieyi_loss: {}".format(str(jieyi_loss_mean)))
    print("csvm_loss: {}".format(str(csvm_loss_mean)))
    print("elas_loss: {}".format(str(elas_loss_mean)))

if __name__ == "__main__":
    data_path = 'synthetic'
    sampleNo = 100

    mu1, mu2 = [0.5, -3], [-0.5, 3]
    sigma1 = sigma2 = [[0.2, 0], [0, 3]]

    noise_ratio = 0.1
    add_noise = True
    if add_noise:

        print("data size: ", 2 * sampleNo + int(2 * sampleNo * noise_ratio))
    else:
        print("data size: ", 2*sampleNo)
    jieyi_C = 0.01
    base_C = 0.005
    elas_C1 = 0.1
    elas_C2 = 0.001

    jieyi = False
    elastic = True
    base = False
    simulation(data_path,
               sampleNo,
               mu1,
               mu2,
               sigma1,
               sigma2,
               add_noise,
               noise_ratio,
               jieyi_C,
               elas_C1,
               elas_C2,
               base_C,
               jieyi=jieyi,
               elastic=elastic,
               base=base

               )



