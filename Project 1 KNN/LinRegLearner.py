import numpy as np

class LinRegLearner(object):

    def __init__(self):       
        self.a1 = None
        self.a2 = None
        self.b = None

    def addEvidence(self,xTrain,yTrain):
        x1 = np.zeros([xTrain.shape[0]])
        x1 = xTrain[:,0]
        x2 = xTrain[:,1]
        A = np.vstack([x1, x2, np.ones(len(x1))]).T
        self.a1, self.a2, self.b = np.linalg.lstsq(A, yTrain)[0]
    
    def query(self,xTest):
        result = np.zeros([xTest.shape[0],xTest.shape[1]+1])
        result[:,0:xTest.shape[1]] = xTest[:,0:xTest.shape[1]]
        for m in range(0,xTest.shape[0]):
            test = xTest[m]
            y = self.a1 * test[0] + self.a2 * test[1] +self.b
            result[m, (xTest.shape[1])] = y

        return result
