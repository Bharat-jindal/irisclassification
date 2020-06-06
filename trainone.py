import math
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def trainone(data,y,rep,alpha,plot):
    m,n=np.shape(data)
    theta=np.zeros((n,1))
    cost = []
    for _ in range(rep):
        #print('Data',data,theta)
        h = np.dot(data,theta)
        vf = np.vectorize(sigmoid)
        h = vf(h)
        c = -(np.dot(np.transpose(np.log(h)),y) + np.dot(np.transpose(np.log(1- h)),1-y))/len(data)
        #print('cost is -> ',c)

        theta = theta - alpha*np.transpose((np.dot(np.transpose(h - y),data)))/len(data)
        #print('New theta',theta)

        cost.append(c[0][0])
    if plot:
        plt.plot(range(rep),cost)
        plt.xlabel('No of iterations')
        plt.ylabel('Cost')

        plt.title('Variation of Cost vs iteration')

        plt.show()
    return theta

