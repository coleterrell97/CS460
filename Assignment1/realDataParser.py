#! /usr/local/bin/python3

import csv
import numpy as np

class realDataSet:
    def __init__(self, fileName, numFeatures, dataTypes, numBins):
        self.numBins = numBins
        self.dataTypes = dataTypes #0 = categorical, 1 = numerical
        self.numFeatures = numFeatures
        self.dataSource = "./data/" + fileName
        self.features = []
        self.binSize = []
        self.__readCSVFile__()
    #end function __init__


    def __readCSVFile__(self):
        dataList = []
        with open(self.dataSource) as dataCSV:
            dataReader = csv.reader(dataCSV)
            for row in dataReader:
                dataList.append(row)
        #store all data as a multidimensional array of floats
        self.features = dataList[0][0:-1]
        self.data=np.array(dataList[1:-1])
    #end function __readCSVFile__


    def discretizeFeatures(self):
        extrema = self.findExtrema()
        self.findBinSizes(extrema)
        for feature in range(0, self.numFeatures):
            if(self.dataTypes[feature] == 1):
                for dataPoint in self.data:
                    featureValue = float(dataPoint[feature])
                    binNumber = 1
                    while featureValue > extrema[feature][0]:
                        featureValue  = featureValue - self.binSize[feature]
                        if(featureValue > extrema[feature][0] and binNumber < self.numBins):
                            binNumber = binNumber + 1
                    dataPoint[feature] = binNumber



    def findBinSizes(self, extrema):
        for feature in range(0, self.numFeatures):
            featureRange = 0
            if(self.dataTypes[feature] == 1):
                featureRange = extrema[feature][1] - extrema[feature][0]
            self.binSize.append(featureRange/self.numBins)

    def findExtrema(self):
        featureExtrema = []
        for feature in range(0, self.numFeatures):
            featureMax = -100.0
            featureMin = 10000000.0
            if(self.dataTypes[feature] == 1):
                for dataPoint in self.data:
                    dataPointFeatureValue = float(dataPoint[feature])
                    if(dataPointFeatureValue > featureMax):
                        featureMax = dataPointFeatureValue
                    if(dataPointFeatureValue < featureMin):
                        featureMin = dataPointFeatureValue
            featureExtrema.append([featureMin,featureMax])
        return featureExtrema

    def discretizeClassLabels(self):
        for dataPoint in self.data:
            if float(dataPoint[11]) < 20:
                dataPoint[11] = 1
            elif float(dataPoint[11]) > 20 and float(dataPoint[11]) < 40:
                dataPoint[11] = 2
            elif float(dataPoint[11]) > 40 and float(dataPoint[11]) < 60:
                dataPoint[11] = 3
            elif float(dataPoint[11]) > 60 and float(dataPoint[11]) < 80:
                dataPoint[11] = 4
            elif float(dataPoint[11]) > 80 and float(dataPoint[11]) < 100:
                dataPoint[11] = 5

    def printData(self):
        for dataPoint in self.data:
            print(dataPoint)
    #end function printData
