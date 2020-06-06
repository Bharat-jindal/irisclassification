import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def checkPerformance(X,Y,totalTheta):
    resultY = []
    for theta in totalTheta:
        #print('Data',data,theta)
        h = np.dot(X,theta)
        vf = np.vectorize(sigmoid)
        h = vf(h)
        resultY.append(h)
    
    count = 0
    for i in range(len(X)):
        maxInd = 0
        maxVal = resultY[0][i][0]
        for j in range(1,len(resultY)):
            if resultY[j][i][0] > maxVal:
                maxInd = j
                maxVal = resultY[j][i][0]
        if Y[i][maxInd] == 1:
            count +=1
    return (count/len(Y))*100
