#! /usr/local/bin/python3
import csv
import numpy as np


class syntheticDataSet:
    def __init__(self, fileName, numFeatures, numBins):
        self.numFeatures = numFeatures
        self.numBins = numBins
        self.dataSource = "./data/" + fileName
        self.__readCSVFile__()
        self.__discretizeFeatures__()

    def __readCSVFile__(self):
        dataList = []
        with open(self.dataSource) as dataCSV:
            dataReader = csv.reader(dataCSV)
            for row in dataReader:
                dataList.append(row)
        self.data=np.array(dataList).astype(float)

    def printData(self):
        print(self.data)

    def __discretizeFeatures__(self):
        max = np.amax(self.data, axis = 0)
        min = np.amin(self.data, axis = 0)
        for feature in range(0,self.numFeatures):
            featureRange= max[feature]-min[feature]
            binSize = featureRange/self.numBins
            for dataPoint in self.data:
                binNumber = 1
                while dataPoint[feature] > min[feature]:
                    dataPoint[feature] = dataPoint[feature] - binSize
                    if(dataPoint[feature] > min[feature]):
                        binNumber = binNumber + 1
                dataPoint[feature] = binNumber



test = syntheticDataSet("synthetic-1.csv", 2, 5)
