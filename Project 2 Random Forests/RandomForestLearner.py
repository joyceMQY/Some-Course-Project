import numpy as np
import math

#"Bagging" is not a required element of this project.
#It has been moved to extra credit.
#Pruning is not a required element of this project.
class RandomForestLearner(object):

    def __init__(self,k=3):   
        self.k = k
        self.forest = None

    def addEvidence(self,xTrain,yTrain):
        data = np.zeros([xTrain.shape[0],xTrain.shape[1]+1])
        data[:,0:xTrain.shape[1]]=xTrain
        data[:,(xTrain.shape[1])]=yTrain[:,0]
        
        forest = []
        for i in range(0, self.k):
            #first, choose random 60%
            baggingData = self.chooseRandom(data)
            tree = self.buildTree(baggingData, 1)
            forest.append(tree)
        
        self.forest = np.array(forest)
        #print "forest", self.k, self.forest

    def chooseRandom(self, data):
        randData = data.copy()
        for i in range(0, int(randData.shape[0] * 0.6)):
            j = np.random.randint(randData.shape[0]-i, size=1)
            temp = randData[i+j,:].copy()
            randData[i+j,:] = randData[i,:]
            randData[i,:] = temp

        return randData[0:int(randData.shape[0] * 0.6), :]
        
                  
    def buildTree(self, data, index):
        if(data.shape[0] == 1):
            return [index, -1, data[0, -1], -1, -1]      
        else:
            leftData = []
            rightData = []

            while(len(leftData) == 0 or len(rightData) == 0):
                leftData = []
                rightData = []
                randomFactor = np.random.randint(2, size = 1)
                factor = randomFactor[0]

                #If you encounter data where all the values for a particular
                #feature are the same (therefore can't split the data)
                #convert the node to be a leaf node with the value being
                #the mean of the Ys.
                equalValue = True
                for i in range(1, data.shape[0]):
                    if(data[i, factor] != data[i-1, factor]):
                        equalValue=False

                if(equalValue):
                    yValue = np.mean(data[:,-1])                  
                    return [index, -1, yValue, -1, -1]

                else:
                    randomSplit = np.random.randint(data.shape[0], size = 2)
                    splitValue = (data[randomSplit[0], factor] + data[randomSplit[1], factor]) / 2.0

                    for i in range(0, data.shape[0]):
                        if(data[i, factor] <= splitValue):
                            leftData.append(data[i, :])
                        else:
                            rightData.append(data[i, :])
                    leftData = np.array(leftData)
                    rightData = np.array(rightData)

            leftTree = self.buildTree(leftData, index+1)
            left = np.array(leftTree)
            left= left.reshape(-1,5)

            rightTree = self.buildTree(rightData, index + left.shape[0] + 1)
            right = np.array(rightTree)
            right= right.reshape(-1,5)

            tree = []
            node = np.array([index, factor+1, splitValue, index+1, index + left.shape[0] + 1])
            node = [node[0], node[1], node[2], node[3], node[4]]
            tree.append(node)

            for row in left:
                lnode = [row[0],row[1],row[2],row[3],row[4]]
                tree.append(lnode)

            for row in right:
                rnode = [row[0],row[1],row[2],row[3],row[4]]
                tree.append(rnode)
            
            return tree
                
    def query(self,xTest):
        result = np.zeros([xTest.shape[0],xTest.shape[1]+1])        
        result[:,0:xTest.shape[1]] = xTest[:,0:xTest.shape[1]]

        for m in range(0,xTest.shape[0]):
            test = xTest[m]

            val = []
            for i in range (0, self.k):
                value = self.findInTree(test, self.forest[i])
                val.append(value)

            value = np.mean(val)
            result[m, xTest.shape[1]] = value

        return result


    def findInTree(self, test, forest):
        forest = np.array(forest)
        factor = forest[0, 1]
        index = 1

        while(factor != -1):
            splitVal = forest[index - 1, 2]
            if(test[factor - 1] <= splitVal):
                index = forest[index - 1, 3]
            else:
                index = forest[index - 1, 4]

            factor = forest[index - 1, 1]

        return forest[index - 1, 2]
