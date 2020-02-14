#! /usr/local/bin/python3
import realDataParser as parser
import math
import numpy as np
import copy
from anytree import Node, RenderTree, Resolver
from anytree.exporter import DotExporter


class decisionTreeReal:
    def __init__(self, numFeatures, numBins, dataFileName):
        self.numFeatures = numFeatures
        self.numBins = numBins
        self.dataFileName = dataFileName
        self.trainingData = parser.realDataSet("Video_Games_Sales_Subset.csv", self.numFeatures, [0,0,0,0,1,1,1,1,1,0,0], self.numBins)
        self.trainingData.discretizeFeatures()
        self.trainingData.discretizeClassLabels()
        self.trainingData.findCategoricalValues()
        self.decisionTree = Node('TOP')
        self.ID3(self.trainingData.data, "", self.trainingData.features, 0, self.decisionTree)
        #self.createModel()

    def ID3(self, dataSet, target_attribute, attributes, depth, parentNode):
            labelDistribution = self.determineClassLabels(dataSet)
            labelMax = -1
            mostCommonLabel = -1
            for label in range(0,len(labelDistribution)):
                if labelDistribution[label] > labelMax:
                    labelMax = labelDistribution[label]
                    mostCommonLabel = label
            #base cases
            if labelDistribution[0] == dataSet.shape[0]:
                leaf = Node("Between 0-20", parent=parentNode)
            elif labelDistribution[1] == dataSet.shape[0]:
                leaf = Node("Between 20-40", parent=parentNode)
            elif labelDistribution[2] == dataSet.shape[0]:
                leaf = Node("Between 40-60", parent=parentNode)
            elif labelDistribution[3] == dataSet.shape[0]:
                leaf = Node("Between 60-80", parent=parentNode)
            elif labelDistribution[4] == dataSet.shape[0]:
                leaf = Node("Between 80-100", parent=parentNode)
            elif attributes == [] or depth == 3:
                if mostCommonLabel == 0:
                    leaf = Node("Between 0-20", parent=parentNode)
                elif mostCommonLabel == 1:
                    leaf = Node("Between 20-40", parent=parentNode)
                elif mostCommonLabel == 2:
                    leaf = Node("Between 40-60", parent=parentNode)
                elif mostCommonLabel == 3:
                    leaf = Node("Between 60-80", parent=parentNode)
                elif mostCommonLabel == 4:
                    leaf = Node("Between 80-100", parent=parentNode)
            #end base cases
            #begin entropy calculation

            else:
                target_attribute = self.findBestSplit(dataSet, labelDistribution, attributes)
                newRoot = Node(attributes[target_attribute], parent=parentNode)
                newAttributes = copy.deepcopy(attributes)
                newAttributes.pop(target_attribute)
                childSets = self.findChildDataSets(dataSet,target_attribute)
                print(childSets)


    def determineClassLabels(self, dataSet):
        labels = [0, 0, 0, 0, 0]
        for dataPoint in dataSet:
            classLabel = int(dataPoint[11])
            if classLabel == 1:
                labels[0]+= 1
            elif classLabel == 2:
                labels[1]+= 1
            elif classLabel == 3:
                labels[2]+= 1
            elif classLabel == 4:
                labels[3]+= 1
            elif classLabel == 5:
                labels[4]+= 1
        return labels

    def calculateEntropy(self, dataSet, labelDistribution):
        entropy = 0
        for label in labelDistribution:
            if(label != 0):
                entropy = entropy + (-((label/dataSet.shape[0]) * math.log(label/dataSet.shape[0],2)))
        return entropy

    def findBestSplit(self, dataSet, labelDistribution, attributes):
        splitInfoGain = np.zeros([1,len(attributes)])
        parentEntropy = self.calculateEntropy(dataSet, labelDistribution)
        for attribute in range(0,len(attributes)): #for each feature
            splitInfoGain[0][attribute] = self.calculateChildAverageEntropy(dataSet, attribute)
            splitInfoGain[0][attribute] = parentEntropy - splitInfoGain[0][attribute]
        bestSplit = np.where(splitInfoGain[0] == np.amax(splitInfoGain[0]))
        return bestSplit[0][0]


    def calculateChildAverageEntropy(self, dataSet, attribute):
        totalExamples = dataSet.shape[0]
        childAverageEntropy = 0
        childSets = self.findChildDataSets(dataSet,attribute)
        for set in childSets:
            labelDistribution = self.determineClassLabels(set)
            if labelDistribution == [0,0,0,0,0]:
                childAverageEntropy = childAverageEntropy
            else:
                childAverageEntropy = childAverageEntropy + (set.shape[0]/totalExamples*(self.calculateEntropy(set, labelDistribution)))
        return childAverageEntropy

    def findChildDataSets(self, dataSet, attribute):
        childDataSets = []
        if self.trainingData.dataTypes[attribute] == 1: #feature is numerical
            for bin in range(1,self.numBins+1):
                childDataSet = []
                for dataPoint in dataSet:
                    if int(dataPoint[attribute]) == bin:
                        childDataSet.append(dataPoint)
                childDataSet = np.array(childDataSet)
                childDataSets.append(childDataSet)
        elif self.trainingData.dataTypes[attribute] == 0: #feature is categorical
            for value in self.trainingData.categoricalValues[attribute]:
                childDataSet = []
                for dataPoint in dataSet:
                    if dataPoint[attribute] == value:
                        childDataSet.append(dataPoint)
                childDataSet = np.array(childDataSet)
                childDataSets.append(childDataSet)
        return childDataSets

    def printTree(self):
        for pre, fill, node in RenderTree(self.decisionTree):
            print("%s%s" % (pre, node.name))

    def createModel(self):
        self.decisionTreeModel = np.empty([self.numBins, self.numBins])
        for i in range(0, self.numBins):
            for j in range(0, self.numBins):
                r = Resolver()
                query = "Feature1/bin " + str(i+1) + "/Feature2/bin " + str(j+1) + "/*"
                try:
                    prediction = r.glob(self.decisionTree, query)[0].name
                except:
                    try:
                        query = "Feature2/bin " + str(j+1) + "/Feature1/bin " + str(i+1) + "/*"
                        prediction = r.glob(self.decisionTree, query)[0].name
                    except:
                        try:
                            query = "Feature2/bin " + str(j+1) + "/*"
                            prediction = r.glob(self.decisionTree, query)[0].name
                        except:
                            query = "Feature1/bin " + str(i+1) + "/*"
                            prediction = r.glob(self.decisionTree, query)[0].name
                if(prediction == "Label = 1"):
                    prediction = 1
                elif(prediction == "Label = 0"):
                    prediction = 0
                self.decisionTreeModel[i][j] = prediction
