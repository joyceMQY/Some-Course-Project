from KNNLearner import KNNLearner
from LinRegLearner import LinRegLearner
from RandomForestLearner import RandomForestLearner
import csv
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import sys

def readCsvData(filename):
    reader = csv.reader(open(filename, 'rU'), delimiter=',')
    count = 0
    for row in reader:
        count = count + 1
    train = int(count * 0.6)
    Xtrain = np.zeros([train,2])
    Ytrain = np.zeros([train,1])
    Xtest = np.zeros([count-train,2])
    Ytest = np.zeros([count-train,1])

    count = 0
    reader = csv.reader(open(filename, 'rU'), delimiter=',')
    for row in reader:
        if(count < train):
            Xtrain[count,0] = row[0]
            Xtrain[count,1] = row[1]
            Ytrain[count,0] = row[2]
            count = count + 1
        else:
            Xtest[count-train,0] = row[0]
            Xtest[count-train,1] = row[1]
            Ytest[count-train,0] = row[2]
            count = count + 1

    return Xtrain, Ytrain, Xtest, Ytest     

def calRMS(Y, Ytest):
    total = 0
    for i in range(0, len(Y)):
        total = total + (Y[i] - Ytest[i]) * (Y[i] - Ytest[i])

    rms = math.sqrt(total / len(Y))
    return rms

def calCorrcoef(Y, Ytest):
    corr = np.corrcoef(Y, Ytest)
    return corr[0,1]

def createComparisonPlot(xLabel, yLabel, xData, y1Data, y2Data, filename, linename):
    plt.clf()
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(xData, y1Data)
    plt.plot(xData, y2Data)
    plt.legend(linename)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.savefig(filename, format='pdf')
    
def test(filename):

    Xtrain, Ytrain, Xtest, Ytest = readCsvData(filename)
    Y = Ytest[:,0]
    sampleY = Ytrain[:,0]
    bestY = np.zeros([Ytest.shape[0]])
    
    knnTrainTime = np.zeros([100])
    knnQueryTime = np.zeros([100])
    knnCorrelation = np.zeros([100])
    knnRmsError = np.zeros([100])
    kArray = np.zeros([100])
    inSampleRmsErr = np.zeros([100])

    rfTrainTime = np.zeros([100])
    rfQueryTime = np.zeros([100])
    rfCorrelation = np.zeros([100])
    rfRmsError = np.zeros([100])
    
    #KNN Learner and RF Learner, k vary from 1 to 100
    for k in range(1, 101):
        #KNN
        kArray[k-1] = k
        
        learner = KNNLearner(k)
        learner.addEvidence(Xtrain, Ytrain)
        knnTest = learner.query(Xtest)
        knnY = knnTest[:,-1]

        #RMS Error(out-of-sample)
        knnRMS = calRMS(knnY, Y)

        #Correlation Coefficient
        knnCorr = calCorrcoef(knnY, Y)

        knnCorrelation[k-1] = knnCorr
        knnRmsError[k-1] = knnRMS

        #RF
        learner = RandomForestLearner(k)
        learner.addEvidence(Xtrain, Ytrain)
        rfTest = learner.query(Xtest)
        rfY = rfTest[:,-1]

        #RMS Error(out-of-sample)
        rfRMS = calRMS(rfY, Y)

        #Correlation Coefficient
        rfCorr = calCorrcoef(rfY, Y)

        rfCorrelation[k-1] = rfCorr
        rfRmsError[k-1] = rfRMS
        
    linename = ['KNN Learner', 'Random Forest Learner']
    createComparisonPlot('K value', 'RMS Error', kArray, knnRmsError, rfRmsError, 'RMSComparison.pdf', linename)
    linename = ['KNN Learner', 'Random Forest Learner']
    createComparisonPlot('K value', 'Correlation', kArray, knnCorrelation, rfCorrelation, 'CorrComparison.pdf', linename)

if __name__=="__main__":
    filename = sys.argv[1]
    test(filename)
