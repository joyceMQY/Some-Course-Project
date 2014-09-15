import numpy as np
import math

class KNNLearner(object):

    def __init__(self,k=3):       
        self.data = None
        self.k = k

    def addEvidence(self,xTrain,yTrain):
        data = np.zeros([xTrain.shape[0],xTrain.shape[1]+1])
        data[:,0:xTrain.shape[1]]=xTrain
        data[:,(xTrain.shape[1])]=yTrain[:,0]       
        self.data = data
    
    def query(self,xTest):
        k = self.k
        result = np.zeros([xTest.shape[0],xTest.shape[1]+1])
        
        result[:,0:xTest.shape[1]] = xTest[:,0:xTest.shape[1]]
        for m in range(0,xTest.shape[0]):
            dist_data = np.zeros([self.data.shape[0],self.data.shape[1]+1])
            for i in range(0,self.data.shape[0]):
                test = xTest[m]
                train = self.data[i]
                dist_data[i,0:self.data.shape[1]] = self.data[i,0:self.data.shape[1]]
        
                dist = math.sqrt((test[0]-train[0])*(test[0]-train[0]) + (test[1]-train[1])*(test[1]-train[1]))
                dist_data[i,self.data.shape[1]] = dist

            sort = dist_data[dist_data[:,-1].argsort()]
            
            total = 0
            for j in range(0,k):
                total = total + sort[j, self.data.shape[1]-1]
                
            mean = total/k
            result[m, xTest.shape[1]] = mean

        return result
