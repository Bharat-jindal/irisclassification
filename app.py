from trainone import trainone
from readData import readData
from test import checkPerformance
import matplotlib.pyplot as plt

print('Reading the data ...')
[trainX,trainY,testX,testY] = readData()

print('Trainig the data ...')
totalTheta = []
plx = []
ply = []
for iter in range(10,1000,50):
    totalTheta = []
    for i in range(len(trainY[0])):
        theta = trainone(trainX,trainY[:,[i]],iter,0.01,iter == 60 and i==0)
        totalTheta.append(theta)
    plx.append(iter)
    print('Testing the performance for {} itrations ...'.format(iter))
    p = checkPerformance(testX,testY,totalTheta)
    ply.append(p)

plt.plot(plx,ply)
plt.xlabel('Total of iterations')
plt.ylabel('Performance')

plt.title('Performance vs No of iteration')

plt.show()