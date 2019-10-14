import matplotlib.pyplot as plt

import numpy as np

def main():
    x = [-3,-2,-1,0,1,2,3,4]

    y_jieyi = list(map(lambda a: jieyi(a, 0.5), x))
    y_hinge = list(map(lambda a: hinge(a), x))


    import pdb;pdb.set_trace()

    plt.figure(1)
    plt.plot(x,y_jieyi)

    plt.plot(x,y_hinge)
    plt.show()



def jieyi(x, p):
    if x <= 1:
        y = p*((1-x)**2 + (1-x))
    else:
        y =0
    return y

def hinge(x):
    if x <= 1:
        return 1-x
    else:
        return 0



if __name__ =="__main__":
    main()