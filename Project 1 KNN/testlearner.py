from KNNLearner import KNNLearner
from LinRegLearner import LinRegLearner
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

def createLinePlot(xLabel, yLabel, xData, yData, filename, title):
    plt.clf()
    fig = plt.figure()
    plt.plot(xData, yData)
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.savefig(filename, format='pdf')

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
    
def createScatterPlot(xLabel, yLabel, xData, yData, filename):
    plt.clf()
    fig = plt.figure()
    plt.plot(xData, yData, 'o')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.savefig(filename, format='pdf')
    
def test(filename):
    Xtrain, Ytrain, Xtest, Ytest = readCsvData(filename)
    Y = Ytest[:,0]
    sampleY = Ytrain[:,0]
    bestY = np.zeros([Ytest.shape[0]])
    
    trainTime = np.zeros([50])
    queryTime = np.zeros([50])
    correlation = np.zeros([50])
    rmsError = np.zeros([50])
    kArray = np.zeros([50])
    inSampleRmsErr = np.zeros([50])
    
    #KNN Learner, k vary from 1 to 50
    for k in range(1, 51):
        kArray[k-1] = k
        
        learner = KNNLearner(k)

        knnTrainStime = time.time()
        learner.addEvidence(Xtrain, Ytrain)
        knnTrainEtime = time.time()

        knnQueryStime = time.time()
        knnTest = learner.query(Xtest)
        knnQueryEtime = time.time()
        knnY = knnTest[:,-1]

        #Avg Train Time per Instance
        avgKnnTrainTime = (knnTrainEtime - knnTrainStime)/Xtrain.shape[0]
        #Avg Query Time per Instance
        avgKnnQueryTime = (knnQueryEtime - knnQueryStime)/Xtest.shape[0]

        #RMS Error(out-of-sample)
        knnRMS = calRMS(knnY, Y)

        #In-sample RMS Error
        inSampleTest = learner.query(Xtrain)
        inSampleY = inSampleTest[:,-1]
        insampleRMS = calRMS(inSampleY, sampleY)

        #Correlation Coefficient
        knnCorr = calCorrcoef(knnY, Y)

        trainTime[k-1] = avgKnnTrainTime
        queryTime[k-1] = avgKnnQueryTime
        correlation[k-1] = knnCorr
        rmsError[k-1] = knnRMS
        inSampleRmsErr[k-1] = insampleRMS

        if((filename == 'data-classification-prob.csv') and (k == 27)):
            print k
            bestY = knnY
        elif((filename == 'data-ripple-prob.csv') and (k == 3)):
            print k
            bestY = knnY

    createLinePlot('K value', 'Avg Train Time/Instance', kArray, trainTime, 'traintime.pdf', 'Average Train Time')
    createLinePlot('K value', 'Avg Query Time/Instance', kArray, queryTime, 'querytime.pdf', 'Average Query Time')
    createLinePlot('K value', 'Correlation', kArray, correlation, 'correlation.pdf', 'Correlation Coefficient of Predicted Y versus Actual Y')
    createLinePlot('K value', 'RMS Error', kArray, rmsError, 'rms.pdf', 'RMS Error between Predicted Y versus Actual Y')

    linename = ['Out-of-Sample Data', 'In-Sample Data']
    createComparisonPlot('K value', 'RMS Error', kArray, rmsError, inSampleRmsErr, 'comparison.pdf', linename)

    createScatterPlot('Predicted Y', 'Actual Y', bestY, Y, 'bestK.pdf')

    #Linear Regression Learner
    learner = LinRegLearner()

    linTrainStime = time.time()
    learner.addEvidence(Xtrain, Ytrain)
    linTrainEtime = time.time()

    linQueryStime = time.time()
    linTest = learner.query(Xtest)
    linQueryEtime = time.time()
    linY = linTest[:,-1]

    #Avg Train Time per Instance
    avgLinTrainTime = (linTrainEtime - linTrainStime)/Xtrain.shape[0]
    #Avg Query Time per Instance
    avgLinQueryTime = (linQueryEtime - linQueryStime)/Xtest.shape[0]
    print avgLinTrainTime, avgLinQueryTime

    #RMS Error
    linRMS = calRMS(linY, Y)
    print linRMS

    #Correlation Coefficient
    linCorr = calCorrcoef(linY, Y)
    print linCorr

if __name__=="__main__":
    filename = sys.argv[1]
    test(filename)
