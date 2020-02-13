#! /usr/local/bin/python3
import csv
import numpy as np


#syntheticDataSet
#Description: Contains data and function members for storing and discretizing CSV input data
#such that it can be used to train and test a decision tree.

class syntheticDataSet:
    def __init__(self, fileName, numFeatures, numBins):
        self.numFeatures = numFeatures
        self.numBins = numBins
        self.dataSource = "./data/" + fileName
        self.__readCSVFile__()
        self.__discretizeFeatures__()
    #end function __init__


    def __readCSVFile__(self):
        dataList = []
        with open(self.dataSource) as dataCSV:
            dataReader = csv.reader(dataCSV)
            for row in dataReader:
                dataList.append(row)
        #store all data as a multidimensional array of floats
        self.data=np.array(dataList).astype(float)
    #end function __readCSVFile__


    def printData(self):
        print(self.data)
    #end function printData

    def __discretizeFeatures__(self):
        max = np.amax(self.data, axis = 0)
        min = np.amin(self.data, axis = 0)
        for feature in range(0,self.numFeatures):
            featureRange= max[feature]-min[feature]
            binSize = featureRange/self.numBins
            #round continuous data to a range of values between 1 and 5
            for dataPoint in self.data:
                binNumber = 1
                while dataPoint[feature] > min[feature]:
                    dataPoint[feature] = dataPoint[feature] - binSize
                    if(dataPoint[feature] > min[feature]):
                        binNumber = binNumber + 1
                dataPoint[feature] = binNumber
    #end function __discretizeFeatures__
#End class syntheticDataSet
