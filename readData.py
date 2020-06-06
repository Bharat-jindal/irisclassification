import numpy as np
import random

def readData():
    f = open('./data/iris.data','r')
    data = []
    y=[]
    #Iris-virginica=[]
    arr = []
    for i in f:
        arr.append(i.strip())
    arr.pop(-1)
    for i in arr:
        sp = i.split(',')
        a = list(map(float,sp[:-1]))
        a = [1] + a
        data.append(a)
        if sp[-1] == 'Iris-setosa' :
            y.append([1,0,0])
        
        elif sp[-1] == 'Iris-versicolor' :
            y.append([0,1,0])
        
        else:
            y.append([0,0,1])

    print('Total Data Readed -> ',len(data))
    l = len(data)*9//10
    trainX = np.array(data[:l])
    trainY = np.array(y[:l])
    testX = np.array(data[l:])
    testY = np.array(y[l:])
    print('Train data cases -> ',np.size(trainX,axis=0))
    print('Test data cases -> ',np.size(testX,axis=0))

    return [trainX,trainY,testX,testY]