from RandomForestLearner import RandomForestLearner
import csv
import numpy as np
import pandas as pd
import math
import copy
import QSTK.qstkutil.qsdateutil as du
import datetime as dt
import QSTK.qstkutil.DataAccess as da
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkstudy.EventProfiler as ep
import matplotlib.pyplot as plt

def getOneDayDiff(base):
    return base[20]-base[19]

def getDiffofDiff(base):
    return (base[20]-base[19]) - (base[19]-base[18])

def getStdev(base):
    length = len(base)
    mean = sum(base) / length
    deviation = 0
    for i in base:
        deviation += (i - mean) ** 2
    return (deviation / length) ** 0.5

def getAmplitude(base):
    return max(base)-min(base)

def getFrequency(base):
    mean = sum(base) / len(base)
    frequency = 0
    for i in range(0, 20):
        if base[i]<=mean and base[i+1]>mean:
            frequency = frequency + 1
        elif base[i]<mean and base[i+1]>=mean:
            frequency = frequency + 1
        elif base[i]>=mean and base[i+1]<mean:
            frequency = frequency + 1
        elif base[i]>mean and base[i+1]<=mean:
            frequency = frequency + 1
    return frequency

def getDeltafromMean(base):
    mean = sum(base) / len(base)
    return base[20] - mean

def getTrainData():
    filenames = []
    
    for i in range (0, 100):
        if i < 10:
            filename = 'ML4T-00'+str(i)
        #else:
        #    filename = 'ML4T-0'+str(i)

        filenames.append(filename)
    
    #filenames.append('ML4T-000')
    start = dt.datetime(2010, 9, 13)
    end = dt.datetime(2012, 9, 14)

    timeofday = dt.timedelta(hours=16)
    timestamps = du.getNYSEdays(start, end, timeofday)
    dataobj = da.DataAccess('ML4Trading')

    keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']

    data = dataobj.get_data(timestamps, filenames, keys)
    dataDic = dict(zip(keys, data))

    for key in keys:
        dataDic[key] = dataDic[key].fillna(method='ffill')
        dataDic[key] = dataDic[key].fillna(method='bfill')
        dataDic[key] = dataDic[key].fillna(1.0)
    priceTrain = dataDic['actual_close'].values

    print len(priceTrain)

    Xtrain, Ytrain, Y = train(priceTrain)
    print Xtrain
    print Ytrain

    return Xtrain, Ytrain

def train(data):
    maxI = data.shape[1]
    print maxI

    Xtrain = np.zeros([maxI*(len(data)-25),6])
    Ytrain = np.zeros([maxI*(len(data)-25),1])
    Y = np.zeros([maxI*(len(data)-25),1])

    count = 0
    for i in range (0,maxI):
        print "i=", i
        for j in range (0, len(data)-25):
            base = data[j:j+21, i]
            x1 = getOneDayDiff(base)
            x2 = getDiffofDiff(base)
            x3 = getStdev(base)
            x4 = getAmplitude(base)
            x5 = getFrequency(base)
            x6 = getDeltafromMean(base)

            Xtrain[count, 0] = x1
            Xtrain[count, 1] = x2
            Xtrain[count, 2] = x3 
            Xtrain[count, 3] = x4
            Xtrain[count, 4] = x5
            Xtrain[count, 5] = x6

            Ytrain[count, 0] = (data[j+25,i]-data[j+20,i])/data[j+20,i]
            Y[count, 0] = data[j+20,i]
            count = count + 1

    return Xtrain, Ytrain, Y

def getTestData():
    filenames = []    
    filenames.append('ML4T-130')
    start = dt.datetime(2000, 2, 1)
    end = dt.datetime(2012, 9, 14)

    timeofday = dt.timedelta(hours=16)
    timestamps = du.getNYSEdays(start, end, timeofday)
    dataobj = da.DataAccess('ML4Trading')

    keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']

    data = dataobj.get_data(timestamps, filenames, keys)
    dataDic = dict(zip(keys, data))

    for key in keys:
        dataDic[key] = dataDic[key].fillna(method='ffill')
        dataDic[key] = dataDic[key].fillna(method='bfill')
        dataDic[key] = dataDic[key].fillna(1.0)
    priceTest = dataDic['actual_close'].values

    print len(priceTest)

    Xtest, Y, Ytest = train(priceTest)
    print Xtest
    print Ytest

    return Xtest, Ytest    

def calRMS(Y, Ytest):
    total = 0
    for i in range(0, len(Y)):
        total = total + (Y[i] - Ytest[i]) * (Y[i] - Ytest[i])

    rms = math.sqrt(total / len(Y))
    return rms

def calCorrcoef(Y, Ytest):
    corr = np.corrcoef(Y, Ytest)
    return corr[0,1]

def createScatterPlot(xLabel, yLabel, xData, yData, filename):
    plt.clf()
    fig = plt.figure()
    plt.plot(xData, yData, 'o')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.savefig(filename, format='pdf')

def createComparisonPlot(xLabel, yLabel, xData, y1Data, y2Data, filename, linename):
    plt.clf()
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(xData, y1Data, color = 'blue')
    plt.plot(xData, y2Data, color = 'red')
    plt.legend(linename)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.savefig(filename, format='pdf')

def createComparison5Plot(xLabel, yLabel, xData, y1Data, y2Data, y3Data, y4Data, y5Data, filename, linename):
    plt.clf()
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(xData, y1Data)
    plt.plot(xData, y2Data)
    plt.plot(xData, y3Data)
    plt.plot(xData, y4Data)
    plt.plot(xData, y5Data)
    plt.legend(linename)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.savefig(filename, format='pdf')
    
    
def test():
    Xtrain, Ytrain = getTrainData()
    Xtest, Ytest = getTestData()
    firstX1 = Xtest[0:100, 0]
    firstX2 = Xtest[0:100, 1]
    firstX3 = Xtest[0:100, 2]
    firstX4 = Xtest[0:100, 3]
    firstX5 = Xtest[0:100, 4]
    Y = Ytest[:,0]
    Yzeros = np.zeros([100,1])
    firstY = Y[0:100]
    lastY = Y[len(Y)-100: len(Y)]

    firstDate = np.zeros([100])
    lastDate = np.zeros([100])
    for i in range (0, 100):
        firstDate[i] = i-1
        lastDate[i] = i-1   
    '''
    #RF Learner
    k = 30
    #RF
    learner = RandomForestLearner(k)
    learner.addEvidence(Xtrain, Ytrain)
    print "finish step 1.1"
    rfTest = learner.query(Xtest)
    rfY = rfTest[:,-1]
    print "finish step 1.2"

    

    for i in range (0, rfY.shape[0]):
        rfY[i] = (rfY[i] + 1) * Y[i]
        
    firstRfY = rfY[0:100]
    lastRfY = rfY[len(rfY)-100: len(rfY)]

    print "finish step 2"

    writer = csv.writer(open('YpreC2_1.csv', 'wb'), delimiter=',')
    for i in range(0,rfY.shape[0]):
        row_to_enter = [Y[i], rfY[i]]
        writer.writerow(row_to_enter)

    #RMS Error(out-of-sample)
    rfRMS = calRMS(rfY, Y)

    #Correlation Coefficient
    rfCorr = calCorrcoef(rfY, Y)

    print rfRMS, rfCorr

    writer = csv.writer(open('rrr2_1.csv', 'wb'), delimiter=',')
    row_to_enter = ['rms', rfRMS]
    writer.writerow(row_to_enter)
    row_to_enter = ['corr', rfCorr]
    writer.writerow(row_to_enter)

    writer = csv.writer(open('features2_1.csv', 'wb'), delimiter=',')
    for i in range(0,100):
        row_to_enter = [firstX1[i], firstX2[i], firstX3[i], firstX4[i], firstX5[i]]
        writer.writerow(row_to_enter)

    kArray = np.zeros([100])
    for k in range(0, 100):
        kArray[k] = k+1

    linename = ['Y actual', 'Y predict']
    createComparisonPlot('Days', 'Price', kArray, firstY, firstRfY, 'first100Comparison.pdf', linename)
    linename = ['Y actual', 'Y predict']
    createComparisonPlot('Days', 'Price', kArray, lastY, lastRfY, 'last100Comparison.pdf', linename)
    
    createScatterPlot('Predicted Y', 'Actual Y', rfY, Y, 'scatter plot.pdf')
    '''
    linename = ['One Day Diff','Diff of two Diff', 'Stdev', 'Amplitude', 'Frequency']
    createComparison5Plot('Dates', 'Values', firstDate, firstX1, firstX2, firstX3, firstX4, firstX5, 'first100Features2.pdf', linename)

if __name__ == '__main__':
    test()    




       
        


